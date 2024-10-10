import cv2
import time
import os
from classify_images_pushbullet import classify_and_notify  # Import the new function

# URL of the IP webcam stream
ip_webcam_url = "http://192.0.0.4:8080:<port>/video"  # Replace with the actual URL

# Directory to save captured frames
output_directory = "captured_frames_ip_webcam"
os.makedirs(output_directory, exist_ok=True)

# Initialize the video capture from the IP webcam
cap = cv2.VideoCapture(ip_webcam_url)

frame_count = 0  # Counter to save unique file names

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Create a unique filename for each frame
        img_path = os.path.join(output_directory, f"frame_{frame_count}.jpg")
        cv2.imwrite(img_path, frame)

        # Call the classify_and_notify function from classify_images_pushbullet.py
        classify_and_notify(img_path)

        frame_count += 1  # Increment the frame count

    # Wait for a specified interval (e.g., 10 seconds)
    time.sleep(10)  # Adjust the interval as needed

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
