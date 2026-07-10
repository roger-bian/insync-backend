import cv2
import numpy as np
from argparse import ArgumentParser

from src.movenet_load import (
    load_model,
    preprocess_image,
    predict,
    calculate_score,
    add_frame_text,
    drawing_links,
    drawing_joints
)

model = load_model()


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

    while True:
        ret, frame = cap.read()
        if ret:
            image = frame.copy()
            
            # Preprocessing the image
            input_image = preprocess_image(image, 256, 256)
            
            # Making prediction
            keypoints = predict(model, input_image)
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
                    image,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4
                )
                frame_text = add_frame_text(frame_resize, "not analysed", color=(0, 0, 255))

            else:
                print(f"FRAME_SCORE{frame_score}, WORST LINK_NAME:{worst_link_name}, WORST LINK SCORE: {worst_link_score}")
                #frame = cv2.flip(frame,0)
                image = drawing_links(people, link_mae, image, linkwidth=6)
                frame_mask = image.copy()
                people = all_people[count]
                frame_mask = drawing_joints(keypoints,
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

    writer.release()


def argparse():
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to the input video file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparse()
    main(args)