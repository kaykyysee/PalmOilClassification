import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import torch

st.set_page_config(layout="wide")

# Define class names and their corresponding colors
class_names = {0: 'Deformed', 1: 'Ripe', 2: 'Raw'}
class_colors = {
    'Deformed': (0, 0, 255),        # Red
    'Ripe': (0, 100, 0),           # Dark Green
    'Raw': (255, 0, 0)             # Blue
}

# Load YOLO model dynamically
@st.experimental_singleton
def load_model(model_path: str, device: str):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.to(device)
    return model

# Inference function
def infer_image(image, model, conf_threshold):
    model.conf = conf_threshold
    results = model(image)
    results.render()  # Render annotations to the image
    return Image.fromarray(results.ims[0])

# Main Streamlit interface
def main():
    st.title('Palm Oil Detection and Counting')
    st.sidebar.title("Settings")

    # Select YOLO weight file
    model_path = st.sidebar.file_uploader("Upload a YOLO model (.pt)", type=['pt'])
    if not model_path:
        st.warning("Please upload a YOLO model to continue.")
        return

    # Device selection
    device_option = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=not torch.cuda.is_available(), index=0)

    # Load model
    model = load_model(model_path, device)

    # Confidence slider
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5)

    # Input selection
    input_type = st.sidebar.radio("Input Type", ['Image', 'Video'])

    if input_type == 'Image':
        # Upload image
        uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Convert image to array for inference
            image_np = np.array(image)

            # Perform inference
            result_image = infer_image(image_np, model, conf_threshold)
            st.image(result_image, caption="Detected Image", use_column_width=True)

    elif input_type == 'Video':
        st.warning("Video input functionality is under development.")

if __name__ == "__main__":
    main()
