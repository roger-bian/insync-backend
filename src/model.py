import os
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MAX_PEOPLE = 6           # MoveNet MultiPose always reports six person slots
KEYPOINTS_PER_PERSON = 17

# The model accepts any input whose sides are multiples of 32, and its accuracy
# depends far more on the aspect ratio it is fed than on the number of pixels.
# Squashing 16:9 into a square roughly doubles both the per-joint error and the
# link-angle error against a MoveNet Thunder reference; scaling the frame up
# instead makes things worse, because the detector starts missing people
# altogether. So keep the source aspect and hold the short side at 256.
INPUT_SHORT_SIDE = 256
SIDE_MULTIPLE = 32

# Second stage. Google never shipped a MultiPose Thunder, so the only way to get
# Thunder's accuracy on more than one person is to run the single-pose model once
# per person, on a crop taken from the box MultiPose already reports. Thunder's
# input is a fixed 256x256, and it was trained on square crops that leave some
# room around the subject, hence the margin on the box.
REFINE_INPUT_SIDE = 256
REFINE_CROP_MARGIN = 1.25

MODEL_URLS = {
    "movenet-multipose-lightning-1":
        "https://tfhub.dev/google/movenet/multipose/lightning/1?tf-hub-format=compressed",
    "movenet-singlepose-thunder-4":
        "https://tfhub.dev/google/movenet/singlepose/thunder/4?tf-hub-format=compressed",
}
MODELS_DIR = Path(os.environ.get(
    "MOVENET_DIR", Path(__file__).resolve().parent.parent / "models"))


def input_shape(frame_width: int, frame_height: int):
    """
    Model input (height, width) that preserves the frame's aspect ratio.

    The short side is pinned to INPUT_SHORT_SIDE and the long side follows the
    frame's aspect, rounded to the multiple of 32 the model requires. A 1280x720
    frame gives 256x448.
    """
    short, long = sorted((frame_width, frame_height))
    scaled = INPUT_SHORT_SIDE * long / short
    long_side = max(SIDE_MULTIPLE,
                    int(round(scaled / SIDE_MULTIPLE)) * SIDE_MULTIPLE)

    if frame_height <= frame_width:
        return INPUT_SHORT_SIDE, long_side
    return long_side, INPUT_SHORT_SIDE


def load_model(name: str):
    """
    Load a MoveNet SavedModel, downloading it once into MODELS_DIR if absent.

    Loaded directly rather than through tensorflow_hub: it is the same
    SavedModel either way, and going direct drops a dependency and keeps the
    model in the project instead of hub's /tmp cache, which a reboot clears.
    """
    model_dir = MODELS_DIR / name
    if not (model_dir / "saved_model.pb").exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        archive = model_dir / "model.tar.gz"
        print(f"Downloading {name} to {model_dir}")
        urllib.request.urlretrieve(MODEL_URLS[name], archive)
        with tarfile.open(archive) as tar:
            try:
                # Refuses members that would escape the destination.
                tar.extractall(model_dir, filter="data")
            except TypeError:       # Python without extraction filters
                tar.extractall(model_dir)
        archive.unlink()
        # The archive ships a world-writable variables/ and a 0700
        # saved_model.pb, so set sane modes rather than inherit those.
        for path in model_dir.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)

    model = tf.saved_model.load(str(model_dir))
    return model.signatures['serving_default']


def crop_square(box, frame_width: int, frame_height: int):
    """
    Square pixel crop (top, left, side) around a normalised person box.

    Square and unclipped so that the person is never distorted: a crop running
    off the edge of the frame is zero-padded instead, which is what MoveNet's
    own cropping does. Returns None for a box too small to be a person.
    """
    ymin, xmin, ymax, xmax = box
    top, bottom = ymin * frame_height, ymax * frame_height
    left, right = xmin * frame_width, xmax * frame_width

    side = max(bottom - top, right - left) * REFINE_CROP_MARGIN
    if side < SIDE_MULTIPLE:
        return None

    center_y, center_x = (top + bottom) / 2, (left + right) / 2
    return (int(round(center_y - side / 2)),
            int(round(center_x - side / 2)),
            int(round(side)))


def crop_frame(frame, top: int, left: int, side: int):
    """
    Cut a square out of a frame, zero-padding whatever falls outside it.
    """
    height, width = frame.shape[:2]
    y0, y1 = max(top, 0), min(top + side, height)
    x0, x1 = max(left, 0), min(left + side, width)
    if y1 <= y0 or x1 <= x0:
        return None

    if (y0, x0, y1, x1) == (top, left, top + side, left + side):
        return frame[y0:y1, x0:x1]      # fully inside, no copy needed

    patch = np.zeros((side, side, frame.shape[2]), dtype=frame.dtype)
    patch[y0 - top:y1 - top, x0 - left:x1 - left] = frame[y0:y1, x0:x1]
    return patch


class Pose:
    def __init__(self, frame_width: int, frame_height: int, refine: bool = True):
        self.input_height, self.input_width = input_shape(frame_width, frame_height)
        self.model = load_model("movenet-multipose-lightning-1")
        self.refiner = load_model("movenet-singlepose-thunder-4") if refine else None

    def __preprocess_image(self, image, size=None):
        """
        Take a frame of a video read through opencv and prepare it for the model.

        OpenCV hands us BGR, MoveNet expects RGB. The resize keeps the frame's
        aspect ratio and does not pad, so keypoints map straight back onto the
        original frame by multiplying with (height, width).
        """
        width, height = size or (self.input_width, self.input_height)
        # Resizing first and swapping channels after is equivalent — resizing is
        # per-channel — and does the channel swap over fewer pixels.
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Cast to an int32 vector and create a batch (input tensor)
        return tf.cast(tf.expand_dims(image, axis=0), dtype=tf.int32)

    def predict(self, input_image):
        """
        Use the model to predict the keypoints given a frame.

        Returns (keypoints, person_scores, boxes):
          keypoints     - (6, 17, 3) of (y, x, confidence), normalised to 0-1
          person_scores - (6,) instance detection score, one per person slot
          boxes         - (6, 4) instance box (ymin, xmin, ymax, xmax), 0-1
        """
        preprocessed = self.__preprocess_image(input_image)
        # Run model inference.
        outputs = self.model(preprocessed)
        # Output is a [1, 6, 56] tensor: 17 keypoints x (y, x, score) = 51,
        # then the instance bounding box (4) and its detection score (1).
        predictions = outputs['output_0'].numpy()[0]
        keypoints = predictions[:, :51].reshape((MAX_PEOPLE, KEYPOINTS_PER_PERSON, 3))
        boxes = predictions[:, 51:55]
        person_scores = predictions[:, 55]
        return keypoints, person_scores, boxes

    def refine(self, frame, keypoints, boxes, active):
        """
        Re-estimate the keypoints of the detected people with SinglePose Thunder,
        one crop at a time, and write them back into keypoints in place.

        Only the slots in `active` are refined, so the cost is one extra
        inference per person actually being scored rather than six.
        """
        if self.refiner is None:
            return keypoints

        height, width = frame.shape[:2]
        for slot in active:
            crop = crop_square(boxes[slot], width, height)
            if crop is None:
                continue
            top, left, side = crop
            patch = crop_frame(frame, top, left, side)
            if patch is None:
                continue

            preprocessed = self.__preprocess_image(
                patch, size=(REFINE_INPUT_SIDE, REFINE_INPUT_SIDE))
            # Output is a [1, 1, 17, 3] tensor of (y, x, score), normalised to
            # the crop, so undo the crop to get back to frame coordinates.
            refined = self.refiner(preprocessed)['output_0'].numpy().reshape(
                KEYPOINTS_PER_PERSON, 3)
            keypoints[slot, :, 0] = (top + refined[:, 0] * side) / height
            keypoints[slot, :, 1] = (left + refined[:, 1] * side) / width
            keypoints[slot, :, 2] = refined[:, 2]

        return keypoints
