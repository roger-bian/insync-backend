import numpy as np

from .definitions import joints, links

DEFAULT_LINK_COLOR = (0, 255, 255)  # yellow in BGR (opencv swaps the channels)


class Joint:
    def __init__(self, id: int, person_id: int):
        self.person_id = person_id
        self.id = id
        self.name = " ".join(joints[id][0:2]).strip()
        self.color = joints[id][2]
        self.is_ignored = joints[id][3]
        self.x = None  # intended to be a float x coordinate
        self.y = None  # intended to be a float y coordinate
        self.confidence = 0.0

    def update(self, x: float, y: float, confidence: float):
        '''
        Inputs : coordinates and confidence score of the joint after detection
        outputs : nothing, the joint is updated in place
        '''
        self.x = x
        self.y = y
        self.confidence = confidence


class Link:
    def __init__(self, id: int, person_id: int):
        self.id = id
        self.person_id = person_id
        self.joint1_id = links[id][1][0]
        self.joint2_id = links[id][1][1]
        self.name = links[id][2]
        self.color = DEFAULT_LINK_COLOR
        self.joints = None
        self.angle = None
        self.similarity_score = None

    def add_joints(self, joint1, joint2):
        self.joints = (joint1, joint2)

    def set_color(self, color):
        self.color = color

    def add_score(self, similarity_score: float):
        self.similarity_score = similarity_score

    def add_angle(self, angle: float):
        self.angle = angle


class Person:
    """
    A person is allocated once and reused for every frame: only the joint
    coordinates, confidences and link angles change as the video plays.
    """

    def __init__(self, id: int, face_ignored: bool):
        self.id = id
        self.face_ignored = face_ignored
        self.joints = [Joint(k, id) for k in joints]
        self.create_links()

        # Fixed for the lifetime of the person, so computed once rather than
        # rebuilt on every frame by the drawing and scoring code.
        self.display_mask = [
            joint.is_ignored if face_ignored else False for joint in self.joints
        ]
        self.scored_joints = [
            joint for joint, hidden in zip(self.joints, self.display_mask) if not hidden
        ]
        # Joint indices of each link's endpoints, for vectorised angle updates.
        self.link_joint1 = np.array([link.joint1_id for link in self.links])
        self.link_joint2 = np.array([link.joint2_id for link in self.links])
        self.angle_vector = np.zeros(len(self.links))

    def create_links(self):
        if self.face_ignored:
            selected = [key for key, val in links.items() if val[3]]  # face mode off
        else:
            selected = list(links)

        self.links = [Link(key, self.id) for key in selected]
        for link in self.links:
            link.add_joints(self.joints[link.joint1_id], self.joints[link.joint2_id])

    def update_joints(self, x_vect, y_vect, conf_vect):
        for joint, x, y, confidence in zip(self.joints, x_vect, y_vect, conf_vect):
            joint.update(x, y, confidence)

    def set_angles(self, angles):
        '''
        Store this frame's link angles, both as a vector for scoring and on
        each link for drawing.
        '''
        self.angle_vector = angles
        for link, angle in zip(self.links, angles):
            link.angle = angle

    def angles(self):
        return self.angle_vector

    def min_confidence(self):
        if not self.scored_joints:
            return 0.0
        return min(joint.confidence for joint in self.scored_joints)

    def mean_confidence(self):
        if not self.scored_joints:
            return 0.0
        return sum(joint.confidence for joint in self.scored_joints) / len(self.scored_joints)
