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

The two pose models are downloaded on first run into `models/`, so the first
invocation needs network access and takes a little longer to start. Later runs
load them from there. Set `MOVENET_DIR` to keep them somewhere else.

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
451 frames, 449 analyzed (100%), mean sync error 20.3 deg
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
| `--display` | _off_ | Open a preview window, refreshed every third frame. `imshow` plus `waitKey` costs several ms a frame, so it is off unless asked for. |
| `--profile` | _off_ | Print a per-stage timing breakdown of the frame loop; see [Per-frame budget](#per-frame-budget). |
| `--no-refine` | _off_ | Skip the Thunder second stage; see [Two-stage pose estimation](#two-stage-pose-estimation). |

Press `q` to stop early when the preview window is open.

## Two-stage pose estimation

Keypoints are estimated in two passes. MoveNet MultiPose Lightning finds the
people and reports a box per person; MoveNet SinglePose Thunder — the more
accurate model, which only ever shipped in a single-person variant — then
re-estimates the keypoints of each detected person from a square crop of that
box. Only the people that will actually be scored are refined, so the cost is one
extra inference per dancer rather than six.

The failure this fixes is specific. On a fast raised-leg move, Lightning collapses
the moving leg onto the standing one, putting both knees and both ankles in the
same place; Thunder, given a crop where the dancer fills the frame, recovers it.
Over `sample.mp4` (451 frames, 902 person-frames):

| | one-stage | two-stage |
|---|---|---|
| frames scored | 420 (93.1%) | 449 (99.6%) |
| person-frames with a collapsed leg | 37 | 18 |
| median worst scored-joint confidence | 0.39 | 0.48 |
| frames whose worst joint falls below the 0.1 gate | 6.9% | 0.4% |
| median per-joint jitter | 6.4 px | 6.4 px |
| inference per frame on CPU, two dancers | 33 ms | 78 ms |

Confidence is the model's own estimate rather than evidence of accuracy — a
collapsed leg is often reported confidently — so the collapsed-leg count is the
row that carries the accuracy claim, verified by eye on the worst disagreements.
The two stages disagree by more than 37 px on 10% of person-frames, and on that
subset the second stage fixes 11 collapsed legs while introducing 1. Elsewhere it
mostly agrees to within a few pixels, which is why jitter — the frame-to-frame
second difference of each joint, measured on dancers re-identified left to right,
since MoveNet's person slots are not stable between frames — comes out unchanged.
The errors the second stage fixes are episodic, not a constant wobble.

Reported mean sync error rises slightly, from 19.7 to 20.3 deg, which is not a
regression. On the 418 frames both configurations score it is unchanged (19.74 vs
19.93 deg). The rise comes from the 31 frames only the two-stage pipeline can
score, which average 25.3 deg: these are the hard raised-leg frames where the
dancers genuinely diverge most, and the one-stage pipeline was discarding them at
the confidence gate rather than scoring them.

Pass `--no-refine` to skip the second stage when throughput matters more.

## GPU performance

Measured on an RTX 2060, the GPU requirements give roughly a 3x faster inference
step (256x448: 32 ms to 9 ms) and 23 to 38 fps end-to-end, with identical
results — CPU and GPU keypoints agree to a median 0.4 px, and the reported sync
error moves by 0.03 deg. Verified with TensorFlow 2.21.0 on CUDA 12.9 /
cuDNN 9.24.

Those figures are for the MultiPose stage on its own, as `--no-refine` runs it.
With the Thunder second stage on, add one 256x256 inference per dancer. Short
bursty inference can also leave the GPU parked at its idle clock — if the numbers
above do not reproduce, check `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm`
before concluding the model is on the CPU.

## Per-frame budget

`--profile` breaks the loop down by stage. Inference used to be about two thirds
of it, the rest going on decoding, drawing and encoding; that other third is now
around an eighth. On `sample.mp4` (720p, two dancers, RTX 2060):

| | one-stage | two-stage |
|---|---|---|
| loop time, before | 11.5 s | 18.9 s |
| loop time, now | 7.6 s | 15.3 s |
| per frame, now | 16.8 ms | 33.9 ms |
| of which inference | 14.2 ms (84%) | 29.7 ms (88%) |
| of which everything else | 2.6 ms (16%) | 4.2 ms (12%) |

Decoding and encoding no longer appear in the budget at all — at 0.04 and
0.02 ms a frame they are queue hand-offs, not work. Both run on their own thread
(`src/pipeline.py`), which is worth roughly 7 ms a frame on its own: OpenCV
releases the GIL inside the mp4v encoder, so a 5-10 ms encode overlaps the model
completely. The constraint that buys is that a frame handed to the encoder must
not be touched again, which is why nothing in the loop reuses a frame buffer even
where it otherwise could.

What remains is preprocessing (1 ms, plus 1.5 ms of crop-and-resize per refined
dancer) and drawing (1.4 ms). The drawing costs were mostly self-inflicted:
approximating each link's ellipse at one degree made it a 360-gon, and OpenCV's
anti-aliased renderer falls off a cliff for thick strokes on small glyphs — the
status line alone cost 0.44 ms at thickness 2 against 0.04 ms at thickness 1. The
blend is now applied only to the box the skeleton occupies, which is exact rather
than approximate: the draw helpers paint opaque colour, so outside that box
`0.2 x + 0.8 x` returns the frame unchanged, verified bit-for-bit.

None of this moves the output. Keypoints before and after differ by at most
1.4e-6 normalised, which is the same spread the pipeline shows between two runs
of identical code — cuDNN does not promise a stable reduction order. Both
configurations report exactly what they did before: 420 frames at 19.7 deg
one-stage, 449 at 20.3 deg two-stage.

Measured and not worth keeping: `cv2.setNumThreads` at 1, 2 and 4 all landed
within noise of the default, so OpenCV's thread pool is left alone.

One caveat on reading these numbers. The 14.2 ms the profiler attributes to
MultiPose is the cost in the running loop, not the 8-9 ms the same call shows in
a tight benchmark loop. That gap is the idle-clock effect above: now that the
loop no longer fills the gaps between inferences with decoding and encoding, the
GPU spends more of its time waiting, and short bursty work does not hold boost
clocks. It is a real cost and the profiler is right to report it, but it means
the stage is not directly comparable to a microbenchmark of the same call.

## Built With
- [Python](https://www.python.org/)
- [TensorFlow](https://tfhub.dev/google/movenet/multipose/lightning/1) - MoveNet MultiPose Lightning person detection
- [TensorFlow](https://tfhub.dev/google/movenet/singlepose/thunder/4) - MoveNet SinglePose Thunder keypoint refinement
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
