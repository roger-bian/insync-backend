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

The pose model is downloaded from TensorFlow Hub on first run, so the first
invocation needs network access and takes a little longer to start.

### Usage

```
python main.py --video {your/video/here}.mp4
```

This writes an annotated `output.mp4` and prints a summary, for example:
```
451 frames, 410 analyzed (91%), mean sync error 19.2 deg
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
