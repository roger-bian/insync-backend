import numpy as np
from rtmlib import RTMDet, RTMPose

from .profile import Profiler

MAX_PEOPLE = 6           # default cap on people scored per frame
KEYPOINTS_PER_PERSON = 17

# RTMDet-nano fine-tuned for single-class person detection, from OpenMMLab's
# RTMPose project (mmpose's projects/rtmpose/rtmdet/person/). OpenMMLab only
# publishes this checkpoint as a .pth; this ONNX export is a third-party
# re-export of that same checkpoint (Apache-2.0). rtmlib downloads and caches
# it like any other checkpoint URL, so no local conversion step is needed.
DET_MODEL_URL = "https://huggingface.co/bukuroo/RTMDet-ONNX/resolve/main/rtmdet-n-person.onnx"
DET_INPUT_SIZE = (320, 320)

# RTMPose-m, the keypoint half of rtmlib's own "balanced" Body preset, fetched
# from OpenMMLab's CDN (rtmlib falls back to a Hugging Face mirror if that
# host is unreachable). COCO-17 output, same joint order as src/definitions.py.
POSE_MODEL_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
POSE_INPUT_SIZE = (192, 256)


class Pose:
    def __init__(self, max_people: int, det_threshold: float, profiler: Profiler = None):
        self.max_people = max_people
        self.det_model = RTMDet(DET_MODEL_URL, model_input_size=DET_INPUT_SIZE,
                                 score_thr=det_threshold,
                                 backend="onnxruntime", device="cpu")
        self.pose_model = RTMPose(POSE_MODEL_URL, model_input_size=POSE_INPUT_SIZE,
                                   backend="onnxruntime", device="cpu")
        self.profiler = profiler or Profiler(False)

    def predict(self, frame):
        """
        Detect people and estimate their keypoints.

        Returns keypoints, shape (N, 17, 3) of (y, x, confidence) in pixel
        coordinates, for the N <= max_people highest-confidence people RTMDet
        found above det_threshold. RTMDet's single-class NMS already returns
        boxes sorted best-score-first, so truncating to max_people keeps the
        best rather than an arbitrary subset.
        """
        boxes = self.det_model(frame)[:self.max_people]
        self.profiler.lap("detect")

        if len(boxes) == 0:
            # RTMPose defaults to a whole-frame box when given none, which
            # would fabricate a person out of an empty frame, so short-circuit
            # instead of calling it.
            self.profiler.lap("pose")
            return np.empty((0, KEYPOINTS_PER_PERSON, 3), dtype=np.float32)

        xy, scores = self.pose_model(frame, bboxes=boxes)
        self.profiler.lap("pose")

        keypoints = np.empty((len(xy), KEYPOINTS_PER_PERSON, 3), dtype=np.float32)
        keypoints[:, :, 0] = xy[:, :, 1]   # y
        keypoints[:, :, 1] = xy[:, :, 0]   # x
        keypoints[:, :, 2] = scores
        return keypoints
