import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background

# Set background image (ensure you have the right path or URL)
set_background()  # Assuming you are using the CSS method for setting the background

# Set title
st.title('Pneumonia Classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('Upload an image:', type=['jpeg', 'jpg', 'png'])

# Load classifier model
model = load_model('./model/pneumonia_classifier.h5')

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

# Display image and classify
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Display classification result
    st.write("## Prediction: {}".format(class_name))
    st.write("### Confidence Score: {}%".format(int(conf_score * 100)))

