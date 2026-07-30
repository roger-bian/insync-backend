import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from src.model import Pose, MAX_PEOPLE
from src.util import (
    calculate_score,
    make_people,
    add_frame_text,
    draw_links,
    draw_joints
)


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

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(args.output,
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not open for writing: {args.output}")

    pose = Pose()
    # Allocated once and reused: the model always reports the same six slots,
    # and only the ones it actually detected are scored on any given frame.
    people = make_people(MAX_PEOPLE, face_ignored=True)
    scale = np.array([height, width, 1.0])

    if args.display:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    count = 0
    scores = []
    try:
        with tqdm(total=frame_count if frame_count > 0 else None,
                  desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Making prediction
                keypoints, person_scores = pose.predict(frame)
                keypoints = keypoints * scale
                active = detect_active(person_scores, args.det_threshold, args.max_people)

                # Calculate scores
                selected, link_mae, frame_score, worst_link_name, \
                    worst_link_score, ignore_frame = calculate_score(
                        people=people,
                        keypoints=keypoints,
                        active=active,
                        conf_threshold=args.conf_threshold
                        )

                # skip frame if too few people or confidence threshold failed
                if ignore_frame:
                    output = add_frame_text(frame, "not analyzed", color=(0, 0, 255))

                else:
                    scores.append(frame_score)
                    # The draw_* helpers annotate in place, so hold on to a clean
                    # copy for the blend rather than compositing over the drawing.
                    original = frame.copy()
                    annotated = draw_links(selected, link_mae, frame, linkwidth=6)
                    annotated = draw_joints(selected,
                                            frame=annotated,
                                            confidence_display=False)

                    output = cv2.addWeighted(src1=original,
                                             alpha=0.20,
                                             src2=annotated,
                                             beta=0.80,
                                             gamma=0)

                    # OpenCV processes BGR images instead of RGB
                    add_frame_text(output, count, color=(0, 255, 0))
                    add_frame_text(output,
                                   f"{len(selected)}p  score: {frame_score:.0f}deg"
                                   f"  worst: {worst_link_name} {worst_link_score:.0f}deg",
                                   color=(0, 255, 0), org=(10, 90), scale=0.7)

                # write video
                writer.write(output)

                if args.display:
                    cv2.imshow("frame", output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                count += 1
                pbar.update(1)
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    analyzed = len(scores)
    print(f"{count} frames, {analyzed} analyzed ({analyzed / max(count, 1):.0%})"
          + (f", mean sync error {np.mean(scores):.1f} deg" if analyzed else ""))


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
                             "confidence. Calibrated on sample.mp4, where the least confident "
                             "joint sits at ~0.30 in the median frame and ~0.11 at the 10th "
                             "percentile: 0.1 drops the tail where a joint is essentially a "
                             "guess, while keeping ~91%% of frames.")
    parser.add_argument("--no-display", dest="display", action="store_false",
                        help="Process without opening a preview window.")
    args = parser.parse_args()
    args.max_people = max(2, min(args.max_people, MAX_PEOPLE))
    return args


if __name__ == "__main__":
    args = argparse()
    main(args)
