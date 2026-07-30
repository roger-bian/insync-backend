import cv2
import tensorflow as tf
import tensorflow_hub as hub

MAX_PEOPLE = 6           # MoveNet MultiPose always reports six person slots
KEYPOINTS_PER_PERSON = 17
INPUT_SIZE = 256         # must be a multiple of 32


class Pose:
    def __init__(self):
        self.model = self.__load_model()

    # Download the model from TF Hub.
    def __load_model(self):
        """
        load model from tensorflow hub
        """
        model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
        return model.signatures['serving_default']

    def __preprocess_image(self, image, new_width=INPUT_SIZE, new_height=INPUT_SIZE):
        """
        Take a frame of a video read through opencv and prepare it for the model.

        OpenCV hands us BGR, MoveNet expects RGB. The resize deliberately
        squashes rather than pads: keypoints then map straight back onto the
        original frame by multiplying with (height, width), no un-padding.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (new_width, new_height))
        # Cast to an int32 vector and create a batch (input tensor)
        return tf.cast(tf.expand_dims(image, axis=0), dtype=tf.int32)

    def predict(self, input_image):
        """
        Use the model to predict the keypoints given a frame.

        Returns (keypoints, person_scores):
          keypoints     - (6, 17, 3) of (y, x, confidence), normalised to 0-1
          person_scores - (6,) instance detection score, one per person slot
        """
        preprocessed = self.__preprocess_image(input_image)
        # Run model inference.
        outputs = self.model(preprocessed)
        # Output is a [1, 6, 56] tensor: 17 keypoints x (y, x, score) = 51,
        # then the instance bounding box (4) and its detection score (1).
        predictions = outputs['output_0'].numpy()[0]
        keypoints = predictions[:, :51].reshape((MAX_PEOPLE, KEYPOINTS_PER_PERSON, 3))
        person_scores = predictions[:, 55]
        return keypoints, person_scores
