import cv2
import numpy as np

from .person import Person

# Worst-first (threshold, BGR colour); the first threshold met wins.
LINK_COLORS = (
    (30, (28, 25, 215)),   # red
    (20, (97, 174, 253)),  # orange
    (5, (50, 255, 212)),   # yellow
    (0, (162, 255, 0)),    # green
)

# A person whose visible joints are this uncertain is not drawn at all.
DRAW_CONFIDENCE_THRESHOLD = 0.1

# Degrees between the vertices approximating a link's ellipse. At one degree
# each link is a 360-gon; at eight it is a 45-gon, which is indistinguishable
# at these sizes and costs less than half as much to fill.
LINK_POLY_DELTA = 8

# Slack around the keypoint box when working out what was drawn: enough for the
# r=14 joint circles and a link ellipse's linkwidth semi-minor axis.
DRAW_PAD = 20


def draw_joints(people, frame, confidence_display):
    """
    Plot the positions of the joints on a frame.
    """
    for person in people:
        if person.mean_confidence() < DRAW_CONFIDENCE_THRESHOLD:
            continue

        for joint, hidden in zip(person.joints, person.display_mask):
            if hidden:
                continue

            x, y = int(joint.x), int(joint.y)
            # Aliased: at r=14 against anti-aliased links the stair-stepping is
            # not visible, and it is three times cheaper over 52 circles a frame.
            cv2.circle(
                img=frame,
                center=(x, y),
                radius=14,
                color=(255, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_8
            )
            cv2.circle(
                img=frame,
                center=(x, y),
                radius=12,
                color=(120, 10, 120),
                thickness=-1,
                lineType=cv2.LINE_8
            )
            if confidence_display:
                #background rectangle for the confidence score display per joint
                cv2.rectangle(
                    img=frame,
                    pt1=(x - 7, y - 15),    # top left corner
                    pt2=(x + 65, y + 4),    # bottom right corner
                    color=(255, 255, 255),
                    thickness=-1,
                    lineType=cv2.LINE_AA
                )
                #confidence score display per joint
                cv2.putText(
                    img=frame,
                    text=f'{round(joint.confidence, 4)}',
                    org=(x - 5, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                    bottomLeftOrigin=False
                )
    return frame


def link_color(mae: float):
    """
    Colour for a link given its mean absolute angular error.
    """
    for threshold, color in LINK_COLORS:
        if mae >= threshold:
            return color
    return LINK_COLORS[-1][1]


def draw_links(people, link_mae, frame, linkwidth: int):
    """
    Plot the line of the links based on a treshold value for color
    """
    for person in people:
        for i, link in enumerate(person.links):
            link.set_color(link_color(link_mae[i]))

            x1, y1 = int(link.joints[0].x), int(link.joints[0].y)
            x2, y2 = int(link.joints[1].x), int(link.joints[1].y)
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            polygon = cv2.ellipse2Poly(
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                axes=(int(length / 2), linkwidth),
                angle=int(link.angle),
                arcStart=0,
                arcEnd=360,
                delta=LINK_POLY_DELTA
            )
            cv2.fillConvexPoly(
                img=frame,
                points=polygon,
                color=link.color,
                lineType=cv2.LINE_AA
            )
    return frame


def draw_bounds(keypoints, active, width: int, height: int):
    """
    Pixel box (y0, y1, x0, x1) enclosing everything the draw_* helpers touch.

    The helpers paint opaque colour and the blend that follows uses fixed
    weights, so outside this box blending returns the frame unchanged.
    Restricting the blend to it is lossless, and turns two full-frame buffer
    passes into two small ones.
    """
    points = keypoints[active][:, :, :2]
    y0 = int(points[:, :, 0].min()) - DRAW_PAD
    y1 = int(points[:, :, 0].max()) + DRAW_PAD
    x0 = int(points[:, :, 1].min()) - DRAW_PAD
    x1 = int(points[:, :, 1].max()) + DRAW_PAD
    return max(y0, 0), min(y1, height), max(x0, 0), min(x1, width)


def add_frame_text(frame, text, color: tuple, org=(10, 50), scale: float = 2,
                   thickness: int = None):
    """
    Add a line of status text to a frame.

    Stroke weight follows the font size by default. That is not only a matter
    of looks: OpenCV's anti-aliased renderer falls off a cliff for thick
    strokes on small glyphs, and the forty-character status line costs 0.44 ms
    at thickness 2 against 0.04 ms at thickness 1.
    """
    if thickness is None:
        thickness = max(1, round(scale))

    return cv2.putText(img=frame,
                       text=f'{text}',
                       org=org,
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=scale,
                       color=color,
                       thickness=thickness,
                       lineType=cv2.LINE_AA,
                       bottomLeftOrigin=False)


def calculate_score(people, keypoints, active, conf_threshold: float):
    """
    Update the roster with this frame's keypoints and score the detected people.
    Returns the active subset alongside the frame's similarity metrics.
    """
    selected = update_people(people, keypoints, active)
    link_mae, frame_score, worst_link_name, worst_link_score, ignore_frame = \
        similarity_scorer(selected, conf_threshold)
    return selected, link_mae, frame_score, worst_link_name, worst_link_score, ignore_frame


def calculate_angle(joint1, joint2):
    """
    Takes two joint objects and returns the angle of 2 seen from 1
    y in opposite direction to conventional.

    Reference implementation of the angle convention; the per-frame path uses
    the vectorised equivalent in update_people.
    """
    delta_x = joint2.x - joint1.x
    delta_y = joint2.y - joint1.y

    # Use arctan2 which handles all quadrants efficiently
    return np.degrees(np.arctan2(delta_y, delta_x)) % 360


def angular_diff(a, b):
    """
    Smallest absolute difference between two angles in degrees, in [0, 180].
    Angles are circular: 359 and 1 are 2 degrees apart, not 358.
    """
    d = np.abs(np.subtract(a, b)) % 360
    return np.minimum(d, 360 - d)


# Pair indices for a given number of people, which changes rarely if ever.
_PAIR_INDICES = {}


def pair_indices(count: int):
    """
    Every unordered pair of people, cached rather than rebuilt each frame.
    """
    pairs = _PAIR_INDICES.get(count)
    if pairs is None:
        pairs = np.triu_indices(count, k=1)
        _PAIR_INDICES[count] = pairs
    return pairs


def make_people(max_people: int, face_ignored: bool):
    """
    Allocate the person roster once, up front. One entry per person slot the
    pose model can report, whether or not that slot is occupied in a given frame.
    """
    return [Person(person_id, face_ignored) for person_id in range(max_people)]


def update_people(people, keypoints, active):
    """
    Write a frame's keypoints into the pre-allocated roster and return the
    subset of people actually detected. Allocates nothing per person.
    """
    selected = []
    for slot in active:
        person = people[slot]
        y_vect = keypoints[slot, :, 0]
        x_vect = keypoints[slot, :, 1]
        person.update_joints(x_vect, y_vect, keypoints[slot, :, 2])

        # One arctan2 over every link at once, rather than a call per link.
        person.set_angles(np.degrees(np.arctan2(
            y_vect[person.link_joint2] - y_vect[person.link_joint1],
            x_vect[person.link_joint2] - x_vect[person.link_joint1],
        )) % 360)
        selected.append(person)

    return selected


def similarity_scorer(people: list, conf_threshold: float):
    """
    Takes list of person objects.
    Returns list of mean absolute error between angles for each link
    Returns overall frame score.

    Synchronization is undefined for fewer than two people, so those frames are
    flagged to be ignored rather than scored.
    """
    if len(people) < 2:
        return None, None, None, None, True

    # checking if in any min confidence score of any person below the threshold
    for person in people:
        if person.min_confidence() <= conf_threshold:
            return None, None, None, None, True

    #Each row: person column: link_id
    stacked_angles = np.vstack([person.angles() for person in people])

    # Mean absolute angular difference over every pair of people. For two
    # people this is exactly |a - b|, so the draw_links colour thresholds keep
    # their existing calibration, and it extends to any number of people.
    left, right = pair_indices(len(people))
    link_mae = angular_diff(stacked_angles[left], stacked_angles[right]).mean(axis=0)

    #Other frame metrics
    frame_score = float(np.mean(link_mae))
    worst_link_id = int(np.argmax(link_mae))
    worst_link_name = people[0].links[worst_link_id].name
    worst_link_score = float(link_mae[worst_link_id])

    return link_mae, frame_score, worst_link_name, worst_link_score, False
