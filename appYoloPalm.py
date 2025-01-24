import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Define class names and their corresponding colors
class_names = {0: 'Deformed', 1: 'Ripe', 2: 'Raw'}
class_colors = {
    'Deformed': (0, 0, 255),        # Red
    'Ripe': (0, 100, 0),           # Dark Green
    'Raw': (255, 0, 0)             # Blue
}

# Load your model
model_path = 'best.pt'  # Adjust model path as needed
model = YOLO(model_path)

st.title('Palm Oil Detection and Counting')
st.write("Upload an image to detect objects and count them.")

# Select mode
mode = st.radio("Choose detection mode:", ('Bunches <= 30', 'Bunches > 30'))

# Set confidence based on mode
if mode == 'Bunches <= 30':
    conf_threshold = 0.6
else:
    conf_threshold = 0.4

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing detection...")

    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model(source=image_cv, conf=conf_threshold, verbose=False)

    # Initialize counters
    Total = 0
    Deformed = 0
    Ripe = 0
    Raw = 0

    # Process detections
    annotated_image = image_cv.copy()
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        cls = boxes.cls.tolist()  # Convert tensor to list

        for i, class_index in enumerate(cls):
            Total += 1
            if class_index == 0:
                Deformed += 1
            elif class_index == 1:
                Ripe += 1
            elif class_index == 2:
                Raw += 1

            # Annotate image with bounding boxes, labels, and order numbers
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            label = class_names[class_index]
            color = class_colors[label]
            label_with_num = f"{label} {Total}"

            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Background for text
            font_scale = 0.5  # Smaller font scale
            font_thickness = 1  # Smaller thickness
            (w, h), _ = cv2.getTextSize(label_with_num, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(annotated_image, (x1, y1 - h - 4), (x1 + w, y1), color, -1)

            # Put label with number
            cv2.putText(annotated_image, label_with_num, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)  # White text

    # Convert OpenCV image back to PIL format
    result_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    # Display results
    st.image(result_image, caption='Detected Image', use_column_width=True)
    st.write(f"Total     : {Total}")
    st.write(f"Ripe      : {Ripe}")
    st.write(f"Raw       : {Raw}")
    st.write(f"Deformed  : {Deformed}")
