🎵 Emotion-Based Music Player using DeepFace and Custom Model
This project is a user-specific emotion-aware music player that detects the user's emotions in real time and dynamically plays songs from YouTube Music based on those emotions.

🔧 How It Works

1. Dataset Creation (UserDataSetModeller)
* Define a list of audio clips and corresponding emotion labels.
* Play each audio clip.
* While the clip is playing, capture webcam frames (2 per second).
* Use DeepFace to detect emotions in the frames.
* Label each frame with the emotion corresponding to the audio clip being played.
* Store the frames and labels to prepare a training and testing dataset.

2. Model Training
* Use the VGG16 pre-trained model as the base.
* Fine-tune the top layers for emotion recognition.
* Add custom classification layers.
* Compile the model with a lower learning rate.
* Apply data augmentation to enhance dataset variability.
* Use callbacks (early stopping, learning rate reduction, etc.).
* Train the model on the captured user-specific dataset.

3. Model Testing
* Use modeltester.py to validate the model's performance.
* Run real-time webcam detection to verify expected emotion predictions.

4. Main Application
* Launches a PyQt GUI integrated with OpenCV for real-time face and emotion tracking.
* Simultaneously launches a Selenium-controlled Chrome window running YouTube Music.
* When a user plays a song:
* Emotions are tracked for 7 seconds.
* If negative emotion is detected, the song is skipped.
* Emotion detection uses both:
* DeepFace model (lightweight)
* Custom-trained model (heavier but more accurate for the user)
* Final emotion = weighted average:
* Custom Model: 70%
* DeepFace: 30%
* The OpenCV window displays only DeepFace-detected emotion (to reduce lag).

5. Voice Commands
* A speech recognition module (Nova) runs on a separate thread.
* Supports voice commands: "play" / "pause" / "stop" / "next".

📌 Notes
* Emotion analysis stops after the initial 7 seconds of the song.
* Future enhancement: Build a system that learns and suggests songs from your library based on emotion patterns.
