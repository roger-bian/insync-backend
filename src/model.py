import os
import tarfile
import urllib.request
from pathlib import Path

import cv2
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

MODEL_URL = "https://tfhub.dev/google/movenet/multipose/lightning/1?tf-hub-format=compressed"
MODEL_DIR = Path(os.environ.get(
    "MOVENET_DIR", Path(__file__).resolve().parent.parent / "models" / "movenet-multipose-lightning-1"))


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


class Pose:
    def __init__(self, frame_width: int, frame_height: int):
        self.input_height, self.input_width = input_shape(frame_width, frame_height)
        self.model = self.__load_model()

    def __load_model(self):
        """
        Load the SavedModel, downloading it once into MODEL_DIR if absent.

        Loaded directly rather than through tensorflow_hub: it is the same
        SavedModel either way, and going direct drops a dependency and keeps the
        model in the project instead of hub's /tmp cache, which a reboot clears.
        """
        if not (MODEL_DIR / "saved_model.pb").exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            archive = MODEL_DIR / "model.tar.gz"
            print(f"Downloading MoveNet MultiPose Lightning to {MODEL_DIR}")
            urllib.request.urlretrieve(MODEL_URL, archive)
            with tarfile.open(archive) as tar:
                try:
                    # Refuses members that would escape the destination.
                    tar.extractall(MODEL_DIR, filter="data")
                except TypeError:       # Python without extraction filters
                    tar.extractall(MODEL_DIR)
            archive.unlink()
            # The archive ships a world-writable variables/ and a 0700
            # saved_model.pb, so set sane modes rather than inherit those.
            for path in MODEL_DIR.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)

        model = tf.saved_model.load(str(MODEL_DIR))
        return model.signatures['serving_default']

    def __preprocess_image(self, image):
        """
        Take a frame of a video read through opencv and prepare it for the model.

        OpenCV hands us BGR, MoveNet expects RGB. The resize keeps the frame's
        aspect ratio and does not pad, so keypoints map straight back onto the
        original frame by multiplying with (height, width).
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_width, self.input_height))
        # Cast to an int32 vector and create a batch (input tensor)
        return tf.cast(tf.expand_dims(image, axis=0), dtype=tf.int32)

    def predict(self, input_image):
        """
        Use the model to predict the keypoints given a frame.

        Returns (keypoints, person_scores):
          keypoints     - (6, 17, 3) of (y, x, confidence), normalised to 0-1
          person_scores - (6,) instance detection score, one per person slot
        """
        preprocessed = self.__preprocess_image(input_image)
        # Run model inference.
        outputs = self.model(preprocessed)
        # Output is a [1, 6, 56] tensor: 17 keypoints x (y, x, score) = 51,
        # then the instance bounding box (4) and its detection score (1).
        predictions = outputs['output_0'].numpy()[0]
        keypoints = predictions[:, :51].reshape((MAX_PEOPLE, KEYPOINTS_PER_PERSON, 3))
        person_scores = predictions[:, 55]
        return keypoints, person_scores
