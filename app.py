import streamlit as st
import numpy as np
import cv2
import imutils
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io
import tempfile
import os

# Load the trained model
model = load_model('handwriting_recognition_model.h5')

# Load LabelBinarizer from where it was saved (assuming you saved it during training)
# For now, I'll assume it's the same as during training:
LB = LabelBinarizer()
LB.fit(
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
     'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])


# Function to sort contours
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


# Function to extract letters from an image
def get_letters(img_path):
    letters = []
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    # Loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1, 32, 32, 1)
            ypred = model.predict(thresh)
            ypred = LB.inverse_transform(ypred)
            letters.append(ypred[0])
    return letters, image


# Function to get the word from letters
def get_word(letters):
    word = "".join(letters)
    return word


# Main function to process an image and predict the word
def predict_handwritten_text(image):
    # Create a temporary file using tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        img_path = tmp_file.name
        image.save(img_path)  # Save the uploaded image to the temporary file

        letters, image = get_letters(img_path)
        word = get_word(letters)

        # Display the predicted word
        st.write("Predicted Word: ", word)

        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)


# Streamlit UI elements
st.title('Handwriting Recognition')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# If the user uploads a file
if uploaded_file is not None:
    # Open and display the image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Convert image to PIL format for further processing
    image = PILImage.open(uploaded_file)

    # Add a button to predict the text
    if st.button("Predict Handwritten Text"):
        predict_handwritten_text(image)

    # Add a button to clear the image
    if st.button("Remove Image"):
        uploaded_file = None
        st.experimental_rerun()
