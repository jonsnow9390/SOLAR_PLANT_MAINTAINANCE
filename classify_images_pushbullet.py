import sys
import tensorflow as tf
from keras import preprocessing
import numpy as np
import os
from pushbullet import Pushbullet

# Set encoding to UTF-8
if sys.version_info.major == 3:
    sys.stdout.reconfigure(encoding='utf-8')

# Load the pre-trained model
model = tf.keras.models.load_model('solar_panel_classifier.keras')

# Define the image dimensions (make sure they match what the model expects)
img_height = 244
img_width = 244

# Define class names (make sure they are the same as during training)
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# Path to the folder where the new images are stored for testing
test_images_folder = 'c:/Users/mdhar/Desktop/PROJECT/test_images/'

# Your Pushbullet API Key
API_KEY = 'o.MIXHZaFJjo2FUJkeiSQUPZVrUFGaUPWu'  # Replace with your actual API key
pb = Pushbullet(API_KEY)

# Retrieve devices and select the first one
devices = pb.devices
if devices:
    device = devices[0]  # Get the first device
    print(f"Device: {device.nickname}, ID: {device.device_iden}")  # Access device_iden correctly
else:
    print("No device found. Please check your Pushbullet account.")
    sys.exit(1)

def classify_and_notify(img_path):
    """Load an image, preprocess it, classify it using the model, and send a notification if necessary."""
    try:
        # Load and preprocess the image
        img = preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
        img_array = preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Get the index of the class with the highest score
        predicted_class_idx = np.argmax(score)

        # Get the predicted class name and confidence
        predicted_label = class_names[predicted_class_idx]
        confidence = 100 * np.max(score)

        print(f"Image: {img_path} | Predicted Class: {predicted_label} | Confidence: {confidence:.2f}%")

        # Custom messages for different predicted classes
        custom_messages = {
            'Bird-drop': "Attention: A bird drop has been detected on your solar panel. Please clean it to ensure optimal performance.",
            'Dusty': "Notice: Dust accumulation is detected on the solar panel. Cleaning is recommended.",
            'Electrical-damage': "Warning: Electrical damage detected. Immediate inspection is advised!",
            'Physical-Damage': "Alert: Physical damage has been identified. Please assess and repair it promptly.",
            'Snow-Covered': "Caution: Your solar panel is covered with snow. Clear it for better energy absorption.",
            'Clean': "Good news! The solar panel is clean and operating efficiently."
        }

        # Check if the predicted class is not 'Clean'
        if predicted_label != 'Clean':
            custom_message = custom_messages.get(predicted_label, "An anomaly has been detected.")
            # Send the image file
            try:
                with open(img_path, "rb") as img_file_obj:
                    # Upload the file
                    file_data = pb.upload_file(img_file_obj, img_path)  # Correctly upload the file
                    
                    # Check if the upload was successful
                    if 'file_url' in file_data:
                        # Push the uploaded file as a notification
                        push_file = pb.push_note(
                            title="Anomaly Detected",
                            body=f"{custom_message}\nSee the attached image: {file_data['file_url']}",
                            device=device  # Use the device object directly
                        )
                        print(f"Image uploaded successfully. URL: {file_data['file_url']}")
                    else:
                        print(f"Failed to upload file. Response: {file_data}")  # Print the response if no file_url
            except Exception as e:
                print(f"Error uploading image {img_path}: {e}")
                return None, 0  # Handle errors gracefully
        return predicted_label, confidence  # Return the class name and confidence score

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None, 0

# Check if the test images folder exists and is not empty
print(f"Looking for images in: {test_images_folder}")
if os.path.exists(test_images_folder):
    image_files = os.listdir(test_images_folder)
    print(f"Found files: {image_files}")

    if len(image_files) > 0:
        for img_file in image_files:
            img_path = os.path.join(test_images_folder, img_file)
            classify_and_notify(img_path)  # Classify and send notification for each image
    else:
        print(f"The directory {test_images_folder} is empty.")
else:
    print(f"The directory {test_images_folder} does not exist.")

