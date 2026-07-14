import cv2
import numpy as np
import time

from .person import Joint, Person


def draw_joints(keypoints, people , frame, confidence_display):
    """
    Plot the positions of the joints on a frame.
    """
    number_people = len(people) # number of people selected by the user
    no_display = people[0].joints_to_not_be_displayed()
    start=time.time()
    for person_id in range(number_people):
        if np.mean(keypoints[person_id,:,2]) < 0.1:
            pass
        else:
            for person in people:
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
    return frame


def draw_links(people, link_mae, frame, linkwidth: int):
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
    return people, link_mae, frame_score , worst_link_name , worst_link_score, ignore_frame


def calculate_angle(joint1: Joint, joint2: Joint):
    """
    Takes two joint objects and returns the angle of 2 seen from 1
    y in opposite direction to conventional
    """
    assert joint1.x is not None and joint1.y is not None, "Joint 1 coordinates are not set."
    assert joint2.x is not None and joint2.y is not None, "Joint 2 coordinates are not set."
    delta_x = joint2.x - joint1.x
    delta_y = joint2.y - joint1.y
    
    # Use arctan2 which handles all quadrants efficiently
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to 0-360 range
    return angle_deg % 360


def data_to_people(keypoints: list, number_of_people:int, face_ignored:bool):
    """
    Returns list of people objects with coordinates, confidence and angles assigned to joints and links.
    """
    #Create list of person objects
    people = []
    keypoints= np.array(keypoints)
    
    for person_id in range(number_of_people):
        #Instantiate person
        person = Person(person_id, face_ignored)
        #Assign all the coordinates and confidence to the person
        person.update_joints(keypoints[person_id,:,1], keypoints[person_id,:,0],keypoints[person_id,:,2])
        person.create_links()
        
        for link in person.links:
            #Calculate angle
            link.add_angle(calculate_angle(link.joints[0], link.joints[1]))
        
        people.append(person)

    return people


def similarity_scorer(people:list, conf_threshold:float):
    """
    Takes list of person objects
    Returns list of mean absolute error between angles for each link
    Returns overall frame score
    """
    number_of_people = len(people)
    number_of_links = len(people[0].links)
    ignore_frame=False

    # checking if in any min confidence score of any person below the threshold
    for person in people:
        if person.min_confidence() < conf_threshold:
            ignore_frame=True
            if ignore_frame==True:
                break

    if number_of_people ==2:
        link_mae =[]
        for link_id in range(number_of_links):
            link_mae.append(abs(people[0].angles()[link_id]- people[1].angles()[link_id]))

    else:
        #Each row: person column: link_id
        angle_list = [people[x].angles() for x in range(number_of_people)]
        stacked_angles= np.vstack(angle_list)
        #Calculate mean of each link_id
        mu = np.mean(stacked_angles,axis=0)
        #Calculate errors
        errors = abs(stacked_angles - mu)
        #Calculate mean absolute error
        link_mae = np.mean(errors, axis =0)

    #Other frame metrics
    frame_score = np.mean(link_mae)
    worst_link_score = max(link_mae)
    worst_link_name = people[0].links[np.argmax(link_mae)].name

    return np.array(link_mae) , frame_score,  worst_link_name , worst_link_score, ignore_frame
