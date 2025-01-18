import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants for Model
MODEL_PATH = "c:/Users/Sanghita/Desktop/Human Pose Prj/Human Pose Estimation Project/graph_opt.pb"
DEMO_IMAGE_PATH = "C:/Users/Sanghita/Desktop/Human Pose Prj/Human Pose Estimation Project/manstand.jpg"

# Body Parts and Pose Pairs for Human Pose Estimation
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Input dimensions for the model
INPUT_WIDTH = 368
INPUT_HEIGHT = 368

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        net = cv2.dnn.readNetFromTensorflow(MODEL_PATH)
        return net
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Pose detection function
@st.cache_data
def pose_detector(_net, frame, threshold=0.2):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # Preprocess the image and pass it through the model
    _net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (INPUT_WIDTH, INPUT_HEIGHT),
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = _net.forward()
    out = out[:, :19, :, :]  # Limit to the first 19 parts

    points = []
    for i in range(len(BODY_PARTS)):
        heat_map = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heat_map)
        x = int((frame_width * point[0]) / out.shape[3])
        y = int((frame_height * point[1]) / out.shape[2])
        points.append((x, y) if conf > threshold else None)

    # Draw points and connections
    for pair in POSE_PAIRS:
        part_from, part_to = pair
        id_from, id_to = BODY_PARTS[part_from], BODY_PARTS[part_to]

        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
            cv2.circle(frame, points[id_from], 3, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, points[id_to], 3, (0, 0, 255), cv2.FILLED)

    return frame

# Streamlit App Interface
st.title("Human Pose Estimation with OpenCV")
st.text("Upload an image or use the demo image for pose estimation.")

# Upload Image or Use Demo
img_file_buffer = st.file_uploader("Upload an Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if img_file_buffer:
    image = np.array(Image.open(img_file_buffer))
else:
    image = np.array(Image.open(DEMO_IMAGE_PATH))

st.subheader("Original Image")
st.image(image, caption="Uploaded Image", use_column_width=True)

# Adjust Detection Threshold
threshold = st.slider("Threshold for Key Point Detection", min_value=0, max_value=100, value=20, step=5) / 100

# Load Model
model = load_model()
if model:
    # Perform Pose Estimation
    output_image = pose_detector(model, image, threshold)

    st.subheader("Pose Estimation Result")
    st.image(output_image, caption="Pose Estimation Output", use_container_width=True)
else:
    st.error("Model could not be loaded. Check the model path and try again.")

    # Real-time Video Processing
if st.button("Use Webcam for Real-Time Pose Estimation"):
    video_capture = cv2.VideoCapture(0)  # Open default camera

    stframe = st.empty()  # Create a Streamlit frame for video output

    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame = cv2.resize(frame, (640, 480))
        pose_frame = pose_detector(model, frame, threshold)
        stframe.image(cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    video_capture.release()
    cv2.destroyAllWindows()

