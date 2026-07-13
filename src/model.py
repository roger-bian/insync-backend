import time
import cv2
import tensorflow as tf
import tensorflow_hub as hub


class Pose:
    def __init__(self):
        self.model = self.__load_model()

    # Download the model from TF Hub.
    def __load_model(self):
        """
        load model from tensorflow hub
        """
        start=time.time()
        model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
        model = model.signatures['serving_default']
        
        return model


    def __preprocess_image(self, image, new_width=256, new_height=256):
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
        return input_image


    def predict(self, input_image):
        """
        Use the model to predict the keypoints given a reshaped input_image.
        """
        preprocessed = self.__preprocess_image(input_image)
        # Run model inference.
        start = time.time()
        outputs = self.model(preprocessed)
        # Output is a [1, 6, 56] tensor that we can reshape
        keypoints = outputs['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        return keypoints