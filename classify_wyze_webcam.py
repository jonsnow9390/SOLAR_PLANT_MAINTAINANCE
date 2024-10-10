import cv2
import time
from classify_images_pushbullet import classify_and_notify  # Import the new function

# URL for the Wyze Cam stream (e.g., RTSP stream)
wyze_cam_url = "rtsp://<username>:<password>@<ip_address>:<port>/live"  # Replace with actual RTSP URL

# Initialize the video capture from Wyze Cam
cap = cv2.VideoCapture(wyze_cam_url)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Save the frame as an image file
        img_path = "current_wyze_frame.jpg"
        cv2.imwrite(img_path, frame)

        # Call the classify_and_notify function from classify_images_pushbullet.py
        classify_and_notify(img_path)

    # Wait for a specified interval (e.g., 10 seconds)
    time.sleep(10)  # Adjust the interval as needed

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
