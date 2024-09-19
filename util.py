from PIL import Image, ImageOps
import numpy as np
import streamlit as st

def classify(image, model, class_names):
    """
    Classifies an image using the provided model and class names.

    Parameters:
        image (PIL.Image.Image): The image to be classified.
        model (tensorflow.keras.Model): The trained model for classification.
        class_names (list): List of class names corresponding to the model's output classes.

    Returns:
        tuple: The predicted class name and confidence score.
    """
    # Resize image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Prepare data for model
    data = np.expand_dims(normalized_image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(data)

    # Handle binary or multi-class classification
    if len(prediction[0]) == 2:  # Assuming binary classification
        index = 0 if prediction[0][0] > 0.95 else 1
    else:  # Multi-class classification
        index = np.argmax(prediction)

    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def set_background():
    """
    Sets the background for the Streamlit app.
    Replace the URL with your actual background image URL.
    """
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("https://example.com/background.jpg");  /* Update this URL */
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
