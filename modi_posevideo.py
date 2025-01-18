import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define body parts and pose pairs
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

# Input width and height for the network
inWidth = 368
inHeight = 368

# Load the pre-trained network
net = cv2.dnn.readNetFromTensorflow("c:/Users/Sanghita/Desktop/Human Pose Prj/Human Pose Estimation Project/graph_opt.pb")

# Threshold for confidence
thres = 0.2

# Load video file
cap = cv2.VideoCapture("c:/Users/Sanghita/Desktop/Human Pose Prj/Human Pose Estimation Project/run1.mp4")

if not cap.isOpened():
    raise FileNotFoundError("Video file could not be loaded. Please check the path.")

def pose_estimation(cap):
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No frame captured. Exiting loop.")
                break

            # Resize the frame for processing
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]

            # Prepare input blob for the network
            net.setInput(cv2.dnn.blobFromImage(frame, 2.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))

            # Perform forward pass to get the output
            out = net.forward()
            out = out[:, :19, :, :]  # Use only the first 19 body parts

            assert len(BODY_PARTS) == out.shape[1]

            points = []

            for i in range(len(BODY_PARTS)):
                heatMap = out[0, i, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                points.append((int(x), int(y)) if conf > thres else None)

            # Draw skeleton
            for pair in POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]

                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]

                if points[idFrom] and points[idTo]:
                    cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                    cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

            # Display the frame
            cv2.imshow('Pose Estimation', frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred during pose estimation: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Run the pose estimation function
pose_estimation(cap)
