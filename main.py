import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from src.model import Pose
from src.util import (
    calculate_score,
    add_frame_text,
    draw_links,
    draw_joints
)

pose = Pose()


def main(args):
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(f"output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"), fps,(width,height))
    
    
    all_people = []
    count = 0

    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if ret:            
                # Making prediction
                keypoints = pose.predict(frame)
                keypoints= np.squeeze(np.multiply(keypoints, [height,width,1]))
                
                # Calculate scores
                people, link_mae, frame_score, worst_link_name, \
                    worst_link_score, ignore_frame = calculate_score(
                        keypoints=keypoints,
                        number_of_people=2,
                        face_ignored=True,
                        conf_threshold=0
                        )
                all_people.append(people)

                # skip frame if confidence threshold failed
                if ignore_frame:
                    frame_resize = cv2.resize(
                        frame,
                        (width, height),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                    frame_text = add_frame_text(frame_resize, "not analyzed", color=(0, 0, 255))

                else:
                    #frame = cv2.flip(frame,0)
                    image = draw_links(people, link_mae, frame, linkwidth=6)
                    frame_mask = image.copy()
                    people = all_people[count]
                    frame_mask = draw_joints(keypoints,
                                                people=people,
                                                frame=frame_mask,
                                                confidence_display=False)

                    frame_superposition = cv2.addWeighted(src1=frame,
                                                        alpha=0.20,
                                                        src2=frame_mask,
                                                        beta=0.80,
                                                        gamma=0)


                    frame_resize = cv2.resize(
                            frame_superposition,
                            (width, height),
                            interpolation=cv2.INTER_LANCZOS4
                    ) 
                    
                    # OpenCV processes BGR images instead of RGB
                    frame_text = add_frame_text(frame_resize, count, color=(0, 255, 0))

                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                cv2.imshow("frame", frame_text)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # write video
                writer.write(frame_text)
                
                count += 1
            else:
                break
            
            pbar.update(1)

    cap.release()
    writer.release()


def argparse():
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to the input video file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparse()
    main(args)