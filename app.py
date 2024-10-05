import os
import numpy as np
import tensorflow as tf
import joblib
import streamlit as st
from PIL import Image
import io

# Load the model
if os.path.exists('resnet50v2_model.keras'):
    print(f"Loading model from {'resnet50v2_model.keras'}...")
    model = tf.keras.models.load_model('resnet50v2_model.keras')
else:
    raise FileNotFoundError(f"Model file not found at {'resnet50v2_model.keras'}")

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

# Function to handle predictions
def predict_image(uploaded_file):
    # Load image
    image = Image.open(uploaded_file)

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

    # Prepare the response
    response = {
        'arousal': arousal_label[0][0],
        'dominance': dominance_label[0][0],
        'continuous_outputs': {name: continuous_pred[0][i] for i, name in enumerate(['effort', 'frustration', 'mental_demand', 'performance', 'physical_demand'])}
    }
    return response

# Streamlit app interface
st.title("ResNet50V2 Cognitive Load Classifier")

# Check if running in API mode
if st.experimental_get_query_params().get("api") == ["true"]:
    # API mode: Expecting file upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Get predictions
        result = predict_image(uploaded_file)

        # Display results
        st.json(result)  # Output the JSON response
else:
    # Regular Streamlit mode
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Get predictions
        result = predict_image(uploaded_file)

        # Display results
        st.write(f"Arousal: {result['arousal']}")
        st.write(f"Dominance: {result['dominance']}")
        for name, score in result['continuous_outputs'].items():
            st.write(f"{name.capitalize()}: {score}")
