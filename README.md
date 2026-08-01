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

**CPU** (`requirements-cpu.txt`) — TensorFlow 2.10, which requires Python 3.10
and runs everywhere, including Apple silicon.
```
pip install -U pip
pip install -r requirements-cpu.txt
```

**GPU** (`requirements-gpu.txt`) — TensorFlow 2.16 or newer with the CUDA 12
libraries bundled as wheels, so no system CUDA install is needed. No Python 3.10
requirement. See [GPU setup](#gpu-setup) for the one extra step needed on Linux.
```
pip install -U pip
pip install -r requirements-gpu.txt
```

The pose model is downloaded on first run into `models/`, so the first
invocation needs network access and takes a little longer to start. Later runs
load it from there.

### GPU setup

The model is a plain SavedModel with no custom ops, so TensorFlow places it on a
GPU automatically — no code change or flag needed. On Linux, though, the CUDA
wheels ship `libcusolver.so.11` without putting it on the loader path, so
TensorFlow fails one `dlopen` and silently falls back to CPU. The only hint is a
generic "Cannot dlopen some GPU libraries" warning; the missing library is named
only under `TF_CPP_MIN_LOG_LEVEL=0`.

Add the wheels' library directories to `LD_LIBRARY_PATH` before running (from
inside the activated virtualenv):
```
export LD_LIBRARY_PATH=$(python -c "import glob, os, site; print(':'.join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib')))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

To make it permanent, append that line to the virtualenv's `bin/activate`. Under
pyenv-virtualenv that does not work — `pyenv activate` never sources
`bin/activate` — so use an activate hook instead, which applies to every env
that has the wheels installed and leaves the others alone:
`~/.pyenv/pyenv.d/activate/cuda-ld-library-path.bash`
```bash
after_activate '
  _cuda_libs=""
  for _cuda_dir in "${prefix}"/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "${_cuda_dir}" ] && _cuda_libs="${_cuda_libs}${_cuda_libs:+:}${_cuda_dir}"
  done
  if [ -n "${_cuda_libs}" ]; then
    echo "export _PYENV_CUDA_OLD_LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH-}\";"
    echo "export LD_LIBRARY_PATH=\"${_cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\";"
  fi
'
```
and the matching `~/.pyenv/pyenv.d/deactivate/cuda-ld-library-path.bash`
```bash
after_deactivate '
  if [ -n "${_PYENV_CUDA_OLD_LD_LIBRARY_PATH+x}" ]; then
    if [ -n "${_PYENV_CUDA_OLD_LD_LIBRARY_PATH}" ]; then
      echo "export LD_LIBRARY_PATH=\"${_PYENV_CUDA_OLD_LD_LIBRARY_PATH}\";"
    else
      echo "unset LD_LIBRARY_PATH;"
    fi
    echo "unset _PYENV_CUDA_OLD_LD_LIBRARY_PATH;"
  fi
'
```

Note that setting `os.environ["LD_LIBRARY_PATH"]` from inside Python does not
work: glibc reads it once at process start, so it has to be in the environment
before the interpreter launches.

Confirm the GPU registered:
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```
The output should list a `GPU:0` device alongside `CPU:0`.

On Apple silicon, `tensorflow-macos` is obsolete from 2.16 onwards — plain
`tensorflow` covers it, with `tensorflow-metal` optional for GPU acceleration.

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

## GPU performance

Measured on an RTX 2060, the GPU requirements give roughly a 3x faster inference
step (256x448: 32 ms to 9 ms) and 23 to 38 fps end-to-end, with identical
results — CPU and GPU keypoints agree to a median 0.4 px, and the reported sync
error moves by 0.03 deg. Verified with TensorFlow 2.21.0 on CUDA 12.9 /
cuDNN 9.24.

Inference is only about two thirds of the per-frame budget, so the rest of the
loop — decoding, drawing and encoding — holds the end-to-end gain well under the
3x seen on inference alone.

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
