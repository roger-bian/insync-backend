# 💃💃 in sync
_Your personal AI synchronization assistant._


Is there an objective way to **mathematically quantify** synchronization in dance?

Using trigonometry-based error tracking to objectively determine areas of greatest performance improvement through frame-by-frame video analysis. 

https://user-images.githubusercontent.com/113004083/206076584-8894fda5-c629-41f0-88f4-abdf5d016330.mp4

_Dance video sourced from Urban Dance Camp YouTube channel._

## Application Frontend
This repository is now a standalone command line tool. The original hosted
version of the project — its Streamlit frontend, FastAPI service and GCP
deployment — is described at https://github.com/xkeeja/insync-frontend.

## Getting Started
### Setup

Navigate to the base level of the repository
```
cd {your/path/here}/insync-backend
```

Install requirements & dependencies. There are two sets — pick one:

**CPU** (`requirements-cpu.txt`) — ONNX Runtime's CPU execution provider,
runs anywhere.
```
pip install -U pip
pip install -r requirements-cpu.txt
```

**GPU** (`requirements-gpu.txt`) — `onnxruntime-gpu` on Linux/Windows (CUDA
execution provider); on Mac, the regular `onnxruntime` wheel already bundles
the CoreML execution provider, so it's the same package as the CPU install.
```
pip install -U pip
pip install -r requirements-gpu.txt
```

The two models (RTMDet-nano for person detection, RTMPose-m for keypoints)
are downloaded on first run and cached by `rtmlib`, so the first invocation
needs network access and takes a little longer to start. Later runs load them
from the cache. The cache location follows `rtmlib`'s own convention
(`TORCH_HOME`, or `XDG_CACHE_HOME`, or `~/.cache/rtmlib` by default) — set one
of those to keep the models somewhere else.

### Usage

```
python main.py --video {your/video/here}.mp4
```

This writes an annotated `output.mp4` and prints a summary, for example:
```
420 frames, 420 analyzed (100%), mean sync error 19.3 deg
```

Each analyzed frame is overlaid with the skeletons, the frame number, the
number of people scored, the mean angular error across all links, and the
worst-scoring link. Links are coloured green through red by how far apart the
dancers' limb angles are. Frames where too few people are detected, or where a
joint is too uncertain to trust, are marked `not analyzed` and left unscored.

| Option | Default | Description |
|---|---|---|
| `--video` | _required_ | Path to the input video file. |
| `--output` | `output.mp4` | Path to write the annotated video to. |
| `--max-people` | `6` | Cap on people scored per frame. People are detected automatically. |
| `--det-threshold` | `0.4` | Minimum detection score for a person to count as detected. |
| `--conf-threshold` | `0.1` | A frame is not analyzed if any scored joint falls below this confidence. |
| `--display` | _off_ | Open a preview window, refreshed every third frame. `imshow` plus `waitKey` costs several ms a frame, so it is off unless asked for. |
| `--profile` | _off_ | Print a per-stage timing breakdown of the frame loop; see [Per-frame budget](#per-frame-budget). |

Press `q` to stop early when the preview window is open.

## Pose estimation: RTMDet + RTMPose-m

Keypoints are estimated in a single top-down pass: RTMDet-nano finds the
people and reports a box per person, then RTMPose-m estimates that person's
17 COCO keypoints from the box. Both are from OpenMMLab's RTMPose project, run
here through [`rtmlib`](https://github.com/Tau-J/rtmlib), a small pure
ONNXRuntime wrapper with no PyTorch/mmcv/mmdet/mmpose dependency.

RTMDet-nano-person is only published by OpenMMLab as a PyTorch checkpoint, so
`src/model.py` points at a third-party ONNX re-export
([`bukuroo/RTMDet-ONNX`](https://huggingface.co/bukuroo/RTMDet-ONNX) on
Hugging Face, Apache-2.0) rather than a first-party build — verified to
produce sane person boxes on `sample.mp4` before adopting it. RTMPose-m comes
straight from OpenMMLab's own CDN.

This replaces an earlier MoveNet MultiPose Lightning + SinglePose Thunder
two-stage pipeline. There is no analogous refinement pass here — RTMPose-m
runs once per detected person — so `--no-refine` was removed along with it.
On `sample.mp4` (420 frames, CPU):
```
420 frames, 420 analyzed (100%), mean sync error 19.3 deg
```
RTMDet's detection score is on a different scale than MoveNet's instance
score, and inheriting MoveNet's old `--det-threshold` default of 0.2 let
through occasional spurious third/fourth "person" boxes — a reflection, a
partial limb at the frame edge — on a video with only two dancers. Those
extras, not RTMPose-m's own joint confidence, were what tripped
`--conf-threshold`'s gate: any one low-confidence extra detection is enough to
mark the whole frame `not analyzed`. Raising the default to `0.4` (verified by
re-running at 0.2/0.3/0.4/0.5 and checking both the analyzed fraction and the
reported sync error) drops the spurious detections and recovers all 420
frames, with the sync error unchanged (19.3 deg at both 0.2 and 0.4 — the two
real dancers score the same either way).

## GPU performance

Not yet benchmarked with this pipeline. `onnxruntime-gpu`'s CUDA execution
provider should accelerate both models on Linux/Windows the same way as any
other ONNX model; on Apple Silicon, the standard `onnxruntime` wheel's bundled
CoreML execution provider should do the same. Numbers here were only measured
CPU-only on an Intel Mac (see [Per-frame budget](#per-frame-budget)).

## Per-frame budget

`--profile` breaks the loop down by stage. Measured on `sample.mp4` (1080p,
two dancers, Intel Core i7-9750H, CPU only):

| stage | ms/frame | % wall |
|---|---|---|
| decode | 0.13 | 0.2% |
| detect (RTMDet-nano) | 19.26 | 28.8% |
| pose (RTMPose-m) | 43.84 | 65.6% |
| score | 0.21 | 0.3% |
| draw | 2.79 | 4.2% |
| encode | 0.03 | 0.1% |
| display/tqdm | 0.43 | 0.6% |
| other | 0.14 | 0.2% |
| **total** | **66.85** | 100.0% |

Pose estimation dominates because it runs once per detected person (two here);
detection runs once per frame regardless of how many people are in it.
Decoding and encoding barely register, for the same reason
as before: both run on their own thread (`src/pipeline.py`) and overlap with
the model instead of bracketing it.

## Built With
- [Python](https://www.python.org/)
- [RTMDet](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/rtmdet/person) - person detection
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) - keypoint estimation
- [rtmlib](https://github.com/Tau-J/rtmlib) - ONNXRuntime wrapper for RTMDet/RTMPose
- [OpenCV](https://opencv.org/) - Video decoding, drawing & encoding
- [NumPy](https://numpy.org/) - Angle & similarity scoring

## Acknowledgements
Inspired by [Kanami](https://www.linkedin.com/in/kanami-oyama-9a666b243/)'s love of dance.

## Team Members
- Kanami Oyama ([GitHub](https://github.com/kanpinpon)) ([LinkedIn](https://www.linkedin.com/in/kanami-oyama-9a666b243/))
- Jaylon Saville ([GitHub](https://github.com/jaysaville)) ([LinkedIn](https://www.linkedin.com/in/jaysaville/))
- Vincent-Victor Rodriguez--Le Roy ([GitHub](https://github.com/Slokem)) ([LinkedIn](https://www.linkedin.com/in/vincent-victor-r-328aa5a8/))

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License
