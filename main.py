
import cv2
from keras.models import load_model
import numpy as np
from keras.layers import DepthwiseConv2D


# Custom DepthwiseConv2D to handle the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)


# Load the model with custom objects
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
model = load_model('keras_model.h5', custom_objects=custom_objects)

# CAMERA can be 0 or 1 based on the default camera of your computer.
camera = cv2.VideoCapture(0)

# Grab the labels from the labels.txt file. This will be used later.
labels = [label.strip() for label in open('labels.txt', 'r').readlines()]

while True:
    # Grab the webcam's image.
    ret, image = camera.read()

    # Check if the frame was successfully grabbed
    if not ret:
        print("Failed to grab frame from camera. Exiting...")
        break

    # Resize the raw image into (224-height, 224-width) pixels.
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    normalized_image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    normalized_image = (normalized_image / 127.5) - 1

    # Have the model predict what the current image is.
    probabilities = model.predict(normalized_image)

    # Get the label with the highest probability
    max_prob = max(probabilities[0])
    if max_prob >= 0.99:
        predicted_label = labels[np.argmax(probabilities)]
        # Put the label text on the image
        cv2.putText(image,
                    predicted_label,
                    (10, 30),  # position at which the text will be displayed
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (0, 255, 0),  # text color (Green in BGR)
                    2,  # thickness of the text
                    cv2.LINE_AA)

    # Show the image with the detected label
    cv2.imshow('Webcam Image', image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
