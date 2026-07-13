import numpy as np
from .definitions import joints, links


class Joint:
    def __init__(self, id :int, person_id : int):
        self.person_id = person_id
        self.id = id
        self.name = " ".join(joints[id][0:2])
        self.color = joints[id][2]
        self.is_ignored = joints[id][3]
        self.x = None # intented to be a float x coordinate
        self.y = None # intented to be a float y coordinate

    def add_coord(self, x : float, y : float):
        '''
        Inputs : coordinate x, y of the joint after detection
        outputs : joint itslef with coordinates updated
        '''
        self.x = x
        self.y = y
        return self

    def add_confidence(self, confidence : float):
        '''
        Inputs : confidence score from pose detection model for this joint
        outputs : joint itself with confidence scores updated
        '''
        # if confidence <= 0.2:
        #     self.bad_confidence = True
        # else:
        #     self.bad_confidence = False
        self.confidence = confidence
        return self


class Link:
    def __init__(self, id:int, person_id :int):
        self.id = id
        self.person_id = person_id
        self.joint1_id = links[id][1][0]
        self.joint2_id = links[id][1][1]
        self.name = links[id][2]
        self.color = "yellow" # by default the color is yellow

    def add_joints(self, joint1, joint2):
        self.joints = (joint1, joint2)
        return self

    def set_color(self, color):
        self.color = color
        return self

    def add_score(self, similarity_score :float):
        self.similarity_score = similarity_score
        return self

    def add_angle(self, angle: float):
        self.angle = angle
        return self


class Person:
    def __init__(self, id:int, face_ignored :bool):
        self.id = id
        self.face_ignored = face_ignored
        self.joints = [Joint(k, id) for k in range(17)]

    def update_joints(self, x_vect, y_vect, conf_vect):
        self.joints = [joint.add_coord(x,y).add_confidence(confidence) \
            for joint ,x, y, confidence in zip(self.joints, x_vect, y_vect, conf_vect)]
        return self

    def create_links(self):
        if self.face_ignored:
            self.links_empty=[]
            for key, val in links.items():
                if val[3]: #filter for face mode off
                    self.links_empty.append(Link(key, self.id))

        else:
            self.links_empty = [Link(k, self.id) for k in range(16)]
        self.links = [link.add_joints(
            self.joints[link.joint1_id],
            self.joints[link.joint2_id]
            ) for link in self.links_empty
        ]
        return self

    def angles(self):
        return [link.angle for link in self.links]

    def joints_to_not_be_displayed(self):
        if self.face_ignored:
            return [joint.is_ignored for joint in self.joints ]
        else:
            return [False for _ in range(17)]

    def min_confidence(self):
        if self.face_ignored:
            list_confidence = []
            for joint in self.joints:
                if joint.is_ignored == False:
                    list_confidence.append(joint.confidence)
            return np.min(list_confidence)
        else:
            return np.min([joint.confidence for joint in self.joints])
