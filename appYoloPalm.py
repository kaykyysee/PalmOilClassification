import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests

# Workaround to prevent OpenCV libGL errors
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

st.set_page_config(layout="wide")

# Define class names and their corresponding colors
class_names = {0: 'Deformed', 1: 'Ripe', 2: 'Raw'}
class_colors = {
    'Deformed': (0, 0, 255),        # Red
    'Ripe': (0, 100, 0),           # Dark Green
    'Raw': (255, 0, 0)             # Blue
}

# Function to load YOLO model
@st.experimental_singleton
def load_model(model_url=None, local_path=None):
    model_path = local_path or "best.pt"
    # If model file does not exist locally, download it
    if not os.path.exists(model_path):
        if model_url:
            with open(model_path, "wb") as f:
                f.write(requests.get(model_url).content)
        else:
            st.error("Model file not found! Please upload or provide a URL.")
            st.stop()
    # Load YOLO model
    return YOLO(model_path)

# Inference function
def infer_image(image, model, conf_threshold):
    model.conf = conf_threshold
    results = model(image)  # Perform inference
    results.render()  # Render annotations to the image
    return Image.fromarray(results.ims[0])

# Main Streamlit app
def main():
    st.title('Palm Oil Detection and Counting')
    st.sidebar.title("Settings")

    # Select YOLO model
    model_source = st.sidebar.radio("Model Source", ["Upload", "From URL"])
    model_url = None
    local_model_path = None

    if model_source == "Upload":
        uploaded_model = st.sidebar.file_uploader("Upload a YOLO model (.pt)", type=['pt'])
        if uploaded_model:
            local_model_path = f"uploaded_{uploaded_model.name}"
            with open(local_model_path, "wb") as f:
                f.write(uploaded_model.read())
    elif model_source == "From URL":
        model_url = st.sidebar.text_input("Enter the model URL")
        if not model_url:
            st.warning("Please provide a URL for the YOLO model.")
            return

    # Load model
    model = load_model(model_url=model_url, local_path=local_model_path)

    # Confidence threshold
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5)

    # Input type selection
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
