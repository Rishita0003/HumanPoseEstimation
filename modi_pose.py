import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants for body parts and pose pairs
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Input dimensions for pose detection
width = 368
height = 368
inWidth = width
inHeight = height

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow("c:/Users/Sanghita/Desktop/Human Pose Prj/Human Pose Estimation Project/graph_opt.pb")

# Threshold for detecting key points
thres = 0.2


def poseDetector(frame):
    """Detect poses and draw keypoints on the frame."""
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    out = net.forward()
    out = out[:, :19, :, :]
    
    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
        
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame


# Load input image
input_image = cv2.imread("C:/Users/Sanghita/Desktop/Human Pose Prj/Human Pose Estimation Project/manstand.jpg")

# Resize the image for consistent processing
input_image_resized = cv2.resize(input_image, (800, 800))
cv2.imwrite("C:/Users/Sanghita/Desktop/Human Pose Prj/Human Pose Estimation Project/Resized-Image.jpg", input_image_resized)

# Convert the resized image to grayscale for testing
input_image_gray = cv2.cvtColor(input_image_resized, cv2.COLOR_BGR2GRAY)
cv2.imwrite("C:/Users/Sanghita/Desktop/Human Pose Prj/Human Pose Estimation Project/Grayscale-Image.jpg", input_image_gray)

# Perform pose detection
output_image = poseDetector(input_image_resized)

# Save the output
cv2.imwrite("C:/Users/Sanghita/Desktop/Human Pose Prj/Human Pose Estimation Project/Output-Image.png", output_image)

# Display transitions using Matplotlib
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original Resized Image
axs[0].imshow(cv2.cvtColor(input_image_resized, cv2.COLOR_BGR2RGB))
axs[0].set_title("Resized Image")
axs[0].axis("off")

# Grayscale Image
axs[1].imshow(input_image_gray, cmap="gray")
axs[1].set_title("Grayscale Image")
axs[1].axis("off")

# Pose Detection Output
axs[2].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
axs[2].set_title("Pose Detection Output")
axs[2].axis("off")

plt.tight_layout()
plt.show()
