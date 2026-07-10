# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from colorama import Fore, Style
import time
import os
import glob

#Import calculation functions
from .calculations import data_to_people, similarity_scorer


# Load the input image.
def load_image(path : str):
    """
    Take the path (as a string)of an image load and prepare it to be ingested by the model
    MoveNet Multipose Lightning 1
    input : path as a string
    output : tensorflow tensor 256 by 256 RGB with tf.int32 values
    """
    image = tf.io.read_file(path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 160, 256), dtype=tf.int32)
    return image

# Download the model from TF Hub.
def load_model():
    """
    load model from tensorflow hub
    """
    start=time.time()
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    model = model.signatures['serving_default']
    print(Fore.BLUE + f"model loads in: {time.time()-start}s" + Style.RESET_ALL)
    
    return model

def preprocess_image(image, new_width, new_height):
    """
    take an frame of a video converted to an image through opencv,
    wth the new_width and new height  for reshaping purpose.
    Based on the image original definition :
    - (480p: 854px by 480px)
    - (720p: 854px by 480px)
    - (1080p: 854px by 480px)
    """
    start = time.time()
    image = cv2.resize(image, (new_width, new_height))
    # Resize to the target shape and cast to an int32 vector
    input_image = tf.cast(tf.image.resize_with_pad(image, new_width, new_height), dtype=tf.int32)
    # Create a batch (input tensor)
    input_image = tf.expand_dims(input_image, axis=0)

    print(Fore.BLUE + f"image processed in: {time.time()-start}s" + Style.RESET_ALL)
    print(input_image.shape)
    return input_image

def predict(model, input_image):
    """
    Use the model to predict the keypoints given a reshaped input_image.
    """
    # Run model inference.
    start = time.time()
    outputs = model(input_image)
    # Output is a [1, 6, 56] tensor that we can reshape
    keypoints = outputs['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    print(Fore.BLUE + f"Prediction and keypoint output in: {time.time()-start}s" + Style.RESET_ALL)
    return keypoints

def drawing_joints(keypoints, people , frame, confidence_display):
    """
    Plot the positions of the joints on a frame.
    """
    number_people = len(people) # number of people selected by the user
    no_display = people[0].joints_to_not_be_displayed()
    start=time.time()
    for person_id in range(number_people):
        print(np.mean(keypoints[person_id,:,2]))
        if np.mean(keypoints[person_id,:,2]) < 0.1:
            pass
        else:
            for person in people:
                print("plotting ", person.id)
                for joint, display_off in zip(person.joints, no_display):
                    if display_off:
                        pass
                    else:
                        x = joint.x
                        y = joint.y
                        conf = round(joint.confidence,4)
                        cv2.circle(
                        img=frame,
                        center=(int(x), int(y)),
                        radius=14,
                        color=(255,255,255),
                        thickness=-1,
                        lineType=cv2.LINE_AA
                        )
                        cv2.circle(
                        img=frame,
                        center=(int(x), int(y)),
                        radius=12,
                        color=(120,10,120),
                        thickness=-1,
                        lineType=cv2.LINE_AA
                        )
                        if confidence_display:
                            X_top_box = int(x)-7
                            Y_top_box = int(y)-15
                            X_bottom_box = int(x)+65
                            Y_bottom_box = int(y)+4


                            #background rectangle for the confidence score display per joint
                            cv2.rectangle(
                                img=frame,
                                pt1=(X_top_box,Y_top_box), # top left corner
                                pt2=(X_bottom_box,Y_bottom_box),#bottom right corner
                                color=(255,255,255),
                                thickness=-1,
                                lineType=cv2.LINE_AA
                            )
                            #confidence score display per joint
                            cv2.putText(
                                img=frame,
                                text=f'{conf}',
                                org=(int(x)-5,int(y)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(0, 0, 0),
                                thickness=1,
                                lineType=cv2.LINE_AA,
                                bottomLeftOrigin=False
                            )
                
    print(Fore.BLUE + f"Plotting joints output made in: {time.time()-start}s" + Style.RESET_ALL)
    return frame

def drawing_links(people, link_mae, frame, linkwidth: int):
    """
    Plot the line of the links based on a treshold value for color
    """
    start=time.time()
    for person in people:
        for i, link in enumerate(person.links):
            mae = link_mae[i]
            if mae>=30:
                link.set_color((28,25,215))# red in BGR channel (opencv swap the channels)
            elif mae>=20:
                link.set_color((97,174,253))
            elif mae>=10:
                link.set_color((50,255,212))
            elif mae>=5:
                link.set_color((50,255,212))

            else:
                link.set_color((162,255,0))

            x1 , y1 = int(link.joints[0].x), int(link.joints[0].y)
            x2 , y2 = int(link.joints[1].x), int(link.joints[1].y)
            X_mean = int((x1+x2)/2)
            Y_mean = int((y1+y2)/2)
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            angle = link.angle
            polygon = cv2.ellipse2Poly(
                center=(X_mean,Y_mean),
                axes=(int(length/2),linkwidth),
                angle= int(angle),
                arcStart=0,
                arcEnd=360,
                delta=1
            )
            cv2.fillConvexPoly(
                img=frame,
                points=polygon,
                color=link.color,
                lineType=cv2.LINE_AA
            )

    print(Fore.BLUE + f"Plotting link output made in: {time.time()-start}s" + Style.RESET_ALL)
    return frame


def add_frame_text(frame, count: int, color:tuple):
    """
    Add frame number to frame
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(img=frame,
                       text=f'{count}',
                       org=(10,50),
                       fontFace=font,
                       fontScale=2,
                       color=color,
                       thickness=2,
                       lineType=cv2.LINE_AA,
                       bottomLeftOrigin=False)


def calculate_score(keypoints , number_of_people:int, face_ignored:bool, conf_threshold:float):
    """
    Calculate the angles between joints given the keypoints.
    Give a similariy score for the the frame.
    """
    start = time.time()
    people =  data_to_people(keypoints , number_of_people, face_ignored)
    link_mae, frame_score, worst_link_name, worst_link_score, ignore_frame = similarity_scorer(people, conf_threshold)
    print(Fore.BLUE + f"Scoring completed in: {time.time()-start}s" + Style.RESET_ALL)
    return people, link_mae, frame_score , worst_link_name , worst_link_score, ignore_frame
