import time

import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from src.model import Pose, MAX_PEOPLE
from src.pipeline import FrameReader, FrameWriter
from src.profile import Profiler
from src.util import (
    calculate_score,
    make_people,
    add_frame_text,
    draw_bounds,
    draw_links,
    draw_joints
)

# The preview costs an X11 round trip and a >=1 ms waitKey, so refresh it a few
# times a second rather than once a frame.
DISPLAY_EVERY = 3


def source_fps(cap):
    """
    Frame rate to give the writer, or 30 if the container does not say.

    Not rounded: truncating 29.97 to 29 makes the output play three percent
    slow against the source.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps):
        return 30.0
    return float(fps)


def detect_active(person_scores, det_threshold: float, max_people: int):
    """
    Person slots the model actually found, best first, capped at max_people.
    """
    active = np.flatnonzero(person_scores >= det_threshold)
    if len(active) > max_people:
        best = np.argsort(person_scores[active])[::-1][:max_people]
        active = active[best]
    return active


def main(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = source_fps(cap)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(args.output,
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not open for writing: {args.output}")

    profiler = Profiler(args.profile)
    pose = Pose(frame_width=width, frame_height=height, refine=args.refine,
                profiler=profiler)
    # Allocated once and reused: the model always reports the same six slots,
    # and only the ones it actually detected are scored on any given frame.
    people = make_people(MAX_PEOPLE, face_ignored=True)
    # Applied in place to the coordinate channels only, so the confidences are
    # left alone and the keypoints stay float32 instead of widening to float64.
    scale = np.array([height, width], dtype=np.float32)

    if args.display:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    # Decoding and encoding both release the GIL inside OpenCV, so they run on
    # their own threads and overlap with the model instead of bracketing it.
    reader = FrameReader(cap)
    sink = FrameWriter(writer)
    reader.start()
    sink.start()

    count = 0
    scores = []
    started = time.perf_counter()
    try:
        with tqdm(total=frame_count if frame_count > 0 else None,
                  desc="Processing frames") as pbar:
            profiler.reset()
            for frame in reader:
                profiler.lap("decode")

                # Making prediction. Only the people that will actually be
                # scored are worth refining, so pick them before the second
                # stage runs and convert to pixels once it has.
                keypoints, person_scores, boxes = pose.predict(frame)
                active = detect_active(person_scores, args.det_threshold, args.max_people)
                keypoints = pose.refine(frame, keypoints, boxes, active)
                keypoints[:, :, :2] *= scale

                # Calculate scores
                selected, link_mae, frame_score, worst_link_name, \
                    worst_link_score, ignore_frame = calculate_score(
                        people=people,
                        keypoints=keypoints,
                        active=active,
                        conf_threshold=args.conf_threshold
                        )
                profiler.lap("score")

                # skip frame if too few people or confidence threshold failed
                if ignore_frame:
                    output = add_frame_text(frame, "not analyzed", color=(0, 0, 255))

                else:
                    scores.append(frame_score)
                    # The draw_* helpers annotate in place, so hold on to a clean
                    # copy for the blend rather than compositing over the drawing
                    # — of the annotated region alone, since the blend leaves
                    # every other pixel exactly as it found it.
                    y0, y1, x0, x1 = draw_bounds(keypoints, active, width, height)
                    region = frame[y0:y1, x0:x1]
                    original = region.copy()

                    draw_links(selected, link_mae, frame, linkwidth=6)
                    draw_joints(selected, frame=frame, confidence_display=False)

                    if original.size:
                        cv2.addWeighted(src1=original,
                                        alpha=0.20,
                                        src2=region,
                                        beta=0.80,
                                        gamma=0,
                                        dst=region)
                    output = frame

                    # OpenCV processes BGR images instead of RGB
                    add_frame_text(output, count, color=(0, 255, 0))
                    add_frame_text(output,
                                   f"{len(selected)}p  score: {frame_score:.0f}deg"
                                   f"  worst: {worst_link_name} {worst_link_score:.0f}deg",
                                   color=(0, 255, 0), org=(10, 90), scale=0.7)
                profiler.lap("draw")

                # write video — the frame belongs to the encoder from here on
                sink.write(output)
                profiler.lap("encode")

                count += 1
                pbar.update(1)

                if args.display and count % DISPLAY_EVERY == 0:
                    cv2.imshow("frame", output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                profiler.lap("display/tqdm")
    finally:
        reader.close()
        sink.close()
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    wall = time.perf_counter() - started

    analyzed = len(scores)
    print(f"{count} frames, {analyzed} analyzed ({analyzed / max(count, 1):.0%})"
          + (f", mean sync error {np.mean(scores):.1f} deg" if analyzed else ""))
    profiler.report(count, wall)


def argparse():
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Path to write the annotated video to.")
    parser.add_argument("--max-people", type=int, default=MAX_PEOPLE,
                        help=f"Cap on people scored per frame (model maximum {MAX_PEOPLE}).")
    parser.add_argument("--det-threshold", type=float, default=0.2,
                        help="Minimum instance score for a person slot to count as detected.")
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                        help="A frame is not analyzed if any scored joint falls below this "
                             "confidence. Calibrated on sample.mp4 against MultiPose alone, "
                             "where the least confident joint sits at ~0.30 in the median "
                             "frame and ~0.11 at the 10th percentile: 0.1 drops the tail "
                             "where a joint is essentially a guess. That gate cost ~7%% of "
                             "frames before the Thunder second stage; with it, under 1%%.")
    parser.add_argument("--display", dest="display", action="store_true",
                        help=f"Open a preview window, refreshed every "
                             f"{DISPLAY_EVERY} frames. Off by default: imshow "
                             f"plus waitKey costs several ms a frame.")
    parser.add_argument("--profile", action="store_true",
                        help="Print a per-stage timing breakdown of the frame loop.")
    parser.add_argument("--no-refine", dest="refine", action="store_false",
                        help="Skip the SinglePose Thunder second stage and keep "
                             "MultiPose Lightning's keypoints as they come.")
    args = parser.parse_args()
    args.max_people = max(2, min(args.max_people, MAX_PEOPLE))
    return args


if __name__ == "__main__":
    args = argparse()
    main(args)
