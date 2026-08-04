import queue
import threading

# Each slot holds a full-resolution frame, so keep them few. Deep enough to
# ride out one slow frame on either side of the loop, shallow enough that the
# decoder does not run far ahead for nothing.
READ_QUEUE_SIZE = 4
WRITE_QUEUE_SIZE = 8

# How often a thread parked on a saturated queue rechecks whether to give up.
POLL_SECONDS = 0.05


class _Worker(threading.Thread):
    """
    A pipeline thread that keeps hold of whatever exception killed it, so the
    loop can re-raise it instead of blocking forever on a queue nobody serves.
    """

    def __init__(self, name: str):
        super().__init__(name=name, daemon=True)
        self.error = None
        self._stopping = threading.Event()

    def run(self):
        try:
            self.work()
        except BaseException as exc:      # noqa: BLE001 - re-raised on the loop
            self.error = exc

    def work(self):
        raise NotImplementedError

    def _check(self):
        if self.error is not None:
            raise self.error


class FrameReader(_Worker):
    """
    Decode frames on a thread and hand them to the loop through a queue.

    OpenCV releases the GIL inside the decoder, so this overlaps decoding with
    inference rather than paying for it between frames. Every read allocates a
    fresh buffer, which is exactly what lets a frame outlive its loop iteration
    and be passed on to FrameWriter without a copy.
    """

    def __init__(self, cap):
        super().__init__("frame-reader")
        self.cap = cap
        self.frames = queue.Queue(maxsize=READ_QUEUE_SIZE)

    def work(self):
        try:
            while not self._stopping.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                if not self._put(frame):
                    return
        finally:
            self._put(None)               # end of stream

    def _put(self, item):
        while not self._stopping.is_set():
            try:
                self.frames.put(item, timeout=POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def __iter__(self):
        while True:
            frame = self.frames.get()
            self._check()
            if frame is None:
                return
            yield frame

    def close(self):
        self._stopping.set()
        self.join(timeout=2.0)


class FrameWriter(_Worker):
    """
    Encode frames on a thread.

    mp4v encoding costs 5-10 ms per 720p frame and OpenCV releases the GIL for
    it, so running it here hides the whole cost behind the model.

    A frame handed to `write` must not be touched again: it is encoded later,
    on another thread. Every frame in the loop comes fresh out of the decoder
    and is annotated before being handed over, so that holds by construction --
    but it is the reason no buffer in the loop is reused.
    """

    def __init__(self, writer):
        super().__init__("frame-writer")
        self.writer = writer
        self.frames = queue.Queue(maxsize=WRITE_QUEUE_SIZE)

    def work(self):
        while True:
            frame = self.frames.get()
            if frame is None:
                return
            self.writer.write(frame)

    def write(self, frame):
        while True:
            self._check()
            if not self.is_alive():
                raise RuntimeError("frame writer stopped unexpectedly")
            try:
                self.frames.put(frame, timeout=POLL_SECONDS)
                return
            except queue.Full:
                continue

    def close(self):
        """Encode whatever is still queued, then stop the thread."""
        if self.is_alive():
            self.frames.put(None)
            self.join()
        self._check()
