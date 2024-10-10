# camera_integration.py

import tensorflow as tf
import numpy as np
import cv2

# Load the pre-trained model
model = tf.keras.models.load_model('solar_panel_classifier.h5')

# Define the image dimensions (make sure they match what the model expects)
img_height = 244
img_width = 244

# Define class names (make sure they are the same as during training)
class_names = ['faulty', 'normal']  # Adjust this based on your dataset

# Function to classify a single frame
def classify_frame(frame):
    img = cv2.resize(frame, (img_height, img_width))
    img_array = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    return class_names[np.argmax(score)], 100 * np.max(score)

# Open the camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break
    
    predicted_label, confidence = classify_frame(frame)
    
    # Display the predicted label and confidence on the frame
    cv2.putText(frame, f'Predicted: {predicted_label} ({confidence:.2f}%)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Show the frame with predictions
    cv2.imshow('Solar Panel Classification', frame)
    
    # Press 'q' to quit the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
