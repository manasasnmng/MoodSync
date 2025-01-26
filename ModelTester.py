import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


# Load your fine-tuned model
model = load_model('3-emotion_recognition_model.h5')

# Define class labels for emotion detection (update these based on your dataset)
emotion_labels = ["disgusted", "happy", "neutral","sad"]  # Add other emotions as needed

# Initialize OpenCV video capture (use 0 for the default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to the correct input size for VGGFace model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (as VGGFace expects RGB)
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by the model
    img = img_to_array(img)  # Convert image to a numpy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize image to [0, 1]
    
    # Make predictions using the fine-tuned model
    predictions = model.predict(img)
    
    # Get the emotion label with the highest prediction probability
    max_index = np.argmax(predictions)
    predicted_emotion = emotion_labels[max_index]
    
    # Display the predicted emotion on the frame
    cv2.putText(frame, predicted_emotion, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame with the emotion label
    cv2.imshow('Emotion Detection', frame)
    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
