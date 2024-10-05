import os
import numpy as np
import tensorflow as tf
import joblib
import streamlit as st
from PIL import Image
import tensorflow as tf

MODEL_SAVE_PATH = 'resnet50v2_model.keras'

#Load the model

if os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")

# Load OneHotEncoders
arousal_encoder = joblib.load('label_encoder_arousal.pkl')
dominance_encoder = joblib.load('label_encoder_dominance.pkl')

# Function to load and preprocess the image
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    img = image.resize((256, 256))  # ResNet50V2 input size
    img_array = np.array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app interface
st.title("ResNet50V2 Cognitive Load Classifier")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model input
    input_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(input_image)
    
    # Extract predictions
    arousal_pred = predictions['arousal_output']
    dominance_pred = predictions['dominance_output']
    continuous_pred = predictions['continuous_output']

    # Get the highest probability class index
    arousal_class_index = np.argmax(arousal_pred, axis=1)
    dominance_class_index = np.argmax(dominance_pred, axis=1)

    # Get the corresponding labels from OneHotEncoders
    arousal_label = arousal_encoder.inverse_transform(np.eye(arousal_encoder.categories_[0].shape[0])[arousal_class_index].reshape(1, -1))
    dominance_label = dominance_encoder.inverse_transform(np.eye(dominance_encoder.categories_[0].shape[0])[dominance_class_index].reshape(1, -1))

    # Display results
    st.write(f"Arousal: {arousal_label[0][0]}")
    st.write(f"Dominance: {dominance_label[0][0]}")

    # Define class names for continuous outputs
    class_names = ['effort', 'frustration', 'mental_demand', 'performance', 'physical_demand']

    # Display continuous output predictions
    for i, name in enumerate(class_names):
        st.write(f"{name.capitalize()}: {continuous_pred[0][i]}")
