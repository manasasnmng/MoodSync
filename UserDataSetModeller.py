import os
import cv2
import time
import threading
import pygame
from deepface import DeepFace

# Directory to save dataset
data_dir = "emotion_dataset"
os.makedirs(data_dir, exist_ok=True)

# Define audio clips and corresponding labels
audio_clips = {
    "neutral": "neutral.mp3",
    "happy":"happy.mp3",
    "sad":"sad.mp3",
    "disgusted":"disgust.mp3"
}

# Ensure subdirectories for each emotion exist
for emotion in audio_clips.keys():
    os.makedirs(os.path.join(data_dir, emotion), exist_ok=True)

def play_audio_and_capture(audio_file, emotion, duration, cap):
    """Plays audio while capturing frames and labeling them."""
    # Initialize audio
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)

    # Start audio playback
    pygame.mixer.music.play()

    # Frame capture settings
    fps = 2  # Frames per second to capture
    interval = 1 / fps  # Interval between captures
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Analyze the face using DeepFace
            analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

            if isinstance(analysis, list) and len(analysis) > 0:
                detected_emotion = analysis[0]["dominant_emotion"]
            elif isinstance(analysis, dict):
                detected_emotion = analysis.get("dominant_emotion")
            else:
                detected_emotion = None
            print(detected_emotion)
            timestamp = int(time.time() * 1000)
            filename = f"{emotion}_{timestamp}.png"
            filepath = os.path.join(data_dir, emotion, filename)
            cv2.imwrite(filepath, frame)
            frame_count += 1

        except Exception as e:
            print(f"Error analyzing frame: {e}")

        time.sleep(interval)

    pygame.mixer.music.stop()
    print(f"Captured {frame_count} frames for emotion: {emotion}")

# Main program
if __name__ == "__main__":
    # Initialize the camera once
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to initialize the camera.")
        exit(1)

    for emotion, audio_file in audio_clips.items():
        print(f"Starting capture for emotion: {emotion}")

        # Create a thread to play audio and capture frames
        capture_thread = threading.Thread(target=play_audio_and_capture, args=(audio_file, emotion, 45, cap))
        capture_thread.start()
        capture_thread.join()

    # Release the camera after all captures
    cap.release()
    print("Data collection complete. Dataset stored in:", data_dir)
