
import sys
import tensorflow as tf
from keras import preprocessing
import numpy as np
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Set encoding to UTF-8 (to avoid any encoding issues with printing)
if sys.version_info.major == 3:
    sys.stdout.reconfigure(encoding='utf-8')

# Load the pre-trained model
model = tf.keras.models.load_model('solar_panel_classifier.keras')

# Define the image dimensions (make sure they match what the model expects)
img_height = 244
img_width = 244

# Define class names (make sure they are the same as during training)
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']  # Adjust this based on your dataset

# Path to the folder where the new images are stored for testing
test_images_folder = 'c:/Users/mdhar/Desktop/PROJECT/test_images/'

# Email sending function
def send_email(predicted_label, image_path):
    """Send an email notification when the panel status is not 'normal' and attach the classified image."""
    try:
        # Email credentials
        sender_email = "johnmooses12@gmail.com"  # Replace with your email
        receiver_email = "20106404.nishasamreen@gmail.com"  # Replace with the receiver's email
        password = "cinh niye ayot hvtc"  # Replace with the App Password generated

        # Email content
        subject = "Solar Panel Status Alert"
        body = f"The solar panel has been classified as '{predicted_label}'. Please check the system immediately."

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach the image
        if os.path.exists(image_path):
            with open(image_path, 'rb') as attachment:
                mime_base = MIMEBase('application', 'octet-stream')
                mime_base.set_payload(attachment.read())
                encoders.encode_base64(mime_base)
                mime_base.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
                msg.attach(mime_base)
        else:
            print(f"Image file '{image_path}' not found.")

        # Set up the server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)

        # Send the email
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)

        print("Email sent successfully with the image attached!")

        # Close the server connection
        server.quit()

    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to classify the image
def classify_image(img_path):
    """Load an image, preprocess it, and classify it using the model."""
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
        
        # Return the class name and the confidence score
        return class_names[predicted_class_idx], 100 * np.max(score)
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
            predicted_label, confidence = classify_image(img_path)
            if predicted_label:
                print(f"Image: {img_file} | Predicted Class: {predicted_label} | Confidence: {confidence:.2f}%")

                # Send email notification if the predicted class is not 'Clean' (assumed normal state)
                if predicted_label != "Clean":  # Adjust 'Clean' to your 'normal' class
                    send_email(predicted_label, img_path)
    else:
        print(f"The directory {test_images_folder} is empty.")
else:
    print(f"The directory {test_images_folder} does not exist.")
       



