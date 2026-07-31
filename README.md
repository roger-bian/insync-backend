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

Requires Python 3.10 (pinned by TensorFlow 2.10).

Navigate to the base level of the repository
```
cd {your/path/here}/insync-backend
```

Install requirements & dependencies
```
pip install -U pip
pip install -r requirements.txt
```

The pose model is downloaded on first run into `models/`, so the first
invocation needs network access and takes a little longer to start. Later runs
load it from there.

### Usage

```
python main.py --video {your/video/here}.mp4
```

This writes an annotated `output.mp4` and prints a summary, for example:
```
451 frames, 420 analyzed (93%), mean sync error 19.7 deg
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
| `--max-people` | `6` | Cap on people scored per frame. People are detected automatically; 6 is the model maximum. |
| `--det-threshold` | `0.2` | Minimum instance score for a person to count as detected. |
| `--conf-threshold` | `0.1` | A frame is not analyzed if any scored joint falls below this confidence. |
| `--no-display` | _off_ | Process without opening a preview window. |

Press `q` to stop early when the preview window is open.

## TODO: TensorFlow upgrade for GPU support

The model is a plain SavedModel with no custom ops, so TensorFlow places it on a
GPU automatically — no code change needed. What blocks it is the `2.10.0` pin:
that wheel needs CUDA 11.2 and cuDNN 8.1 installed system-wide, which it does not
bundle. Measured on an RTX 2060, moving to a modern TensorFlow gives roughly a
3x faster inference step (256x448: 32 ms to 9 ms) and 23 to 38 fps end-to-end,
with identical results — CPU and GPU keypoints agree to a median 0.4 px, and the
reported sync error moves by 0.03 deg.

Steps, in order:

1. Replace all three `tensorflow` lines in `requirements.txt` with
   `tensorflow[and-cuda]` (2.16 or newer ships the CUDA 12 libraries as wheels,
   so no system CUDA install is needed). Verified working with 2.21.0 on
   CUDA 12.9 / cuDNN 9.24.
2. Drop the `numpy<2` pin, which only exists for TensorFlow 2.10.
3. Note that `tensorflow-macos` is obsolete from 2.16 onwards — plain
   `tensorflow` covers Apple silicon, with `tensorflow-metal` optional for GPU
   acceleration there. Confirm before changing this for Mac users.
4. Relax the Python 3.10 requirement in [Setup](#setup); it exists only because
   of the 2.10 pin.
5. Expect to set `LD_LIBRARY_PATH` to the `nvidia/*/lib` directories inside the
   virtualenv's `site-packages`. The wheels ship `libcusolver.so.11` but nothing
   puts it on the loader path, so TensorFlow fails one `dlopen` and silently
   falls back to CPU. The only hint is a generic "Cannot dlopen some GPU
   libraries" warning; the missing library is named only under
   `TF_CPP_MIN_LOG_LEVEL=0`. Confirm the GPU registered with
   `python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"`.

Worth knowing before spending the effort: inference is only about two thirds of
the per-frame budget, so the rest of the loop — decoding, drawing and encoding —
limits the end-to-end gain to well under the 3x seen on inference alone.

## Built With
- [Python](https://www.python.org/)
- [TensorFlow](https://tfhub.dev/google/movenet/multipose/lightning/1) - MoveNet MultiPose Lightning pose detection
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
