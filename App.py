import sys
import cv2
import time
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextBrowser, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap,QFont
from PyQt5.QtCore import QTimer, Qt
from deepface import DeepFace
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import numpy as np
from tensorflow.keras.models import load_model
import speech_recognition as sr
import pyttsx3
import openpyxl
import qdarkstyle  
import random
class YouTubeMusicEmotionAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Music Emotion Analyzer")
        self.setGeometry(100, 100, 800, 600)

        # Initialize UI components
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("border: 1px solid #555; padding: 5px; background-color: #1e1e1e;")

        self.text_browser = QTextBrowser(self)
        self.text_browser.setFont(QFont("Arial", 10))
        self.text_browser.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; padding: 10px;")

        # Arrange components in a vertical layout
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        layout.addWidget(self.camera_label, stretch=3)
        layout.addWidget(self.text_browser, stretch=1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.text_browser.append("Error: Unable to access the camera.")
            return

        # Timer for updating the camera feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_feed)
        self.timer.start(30)  # Update every 30ms

        # Selenium setup
        self.browser = self.setup_browser()
        self.playback_info = {}
        self.last_song_title = ""
        self.emotion_analysis_done = False

        # Start a separate thread for monitoring YouTube Music
        self.monitor_thread = threading.Thread(target=self.monitor_youtube_music, daemon=True)
        self.monitor_thread.start()
        # Voice Assistant
        self.voice_assistant = NovaVoiceAssistant(self.text_browser, self.browser)
        threading.Thread(target=self.voice_assistant.listen, daemon=True).start()
    def setup_browser(self):
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)

    def inject_js(self):
        js_code = """
        (function() {
            console.log("JavaScript is running.");
            window.playbackinfo = {};
            let lastSongTitle = ""; 

            const getPlaybackInfo = function() {
                const songTitle = document.querySelector('ytmusic-player-bar .title');
                const progressBar = document.querySelector('tp-yt-paper-slider#progress-bar'); 

                if (songTitle && songTitle.textContent && progressBar) {
                    if (songTitle.textContent !== lastSongTitle) {
                        lastSongTitle = songTitle.textContent;
                        progressBar.setAttribute('aria-valuenow', 0);
                        console.log("New song detected:", songTitle.textContent);
                    }

                    const progressData = (function() {
                        return progressBar ? {
                            current: parseFloat(progressBar.getAttribute('aria-valuenow')) || 0,
                            max: parseFloat(progressBar.getAttribute('aria-valuemax')) || 1
                        } : null;
                    })();

                    if (progressData) {
                        window.playbackinfo = {
                            songTitle: songTitle.textContent,
                            progress: progressData.current,
                            max: progressData.max
                        };

                        console.log("Song info:", window.playbackinfo);
                    }
                }
            };

            getPlaybackInfo();
            setInterval(getPlaybackInfo, 2000);
        })();
        """
        self.browser.execute_script(js_code)

    def monitor_youtube_music(self):
        self.browser.get("https://music.youtube.com/") 
        time.sleep(5)
        self.inject_js()

        while True:
            try:
                playback_info = self.browser.execute_script("return window.playbackinfo;")
                if playback_info:
                    song_title = playback_info.get("songTitle", "")
                    progress = playback_info.get("progress", 0)
                    max_progress = playback_info.get("max", 1)

                    if song_title != self.last_song_title:
                        self.last_song_title = song_title
                        self.emotion_analysis_done = False
                        self.text_browser.append(f"Now playing: {song_title}")

                        # Perform emotion analysis in a separate thread
                        threading.Thread(target=self.perform_emotion_analysis, args=(7,e)).start()

            except Exception as e:
                self.text_browser.append(f"Error: {e}")
            time.sleep(1)


    def superplay():
        pass
    def perform_emotion_analysis(self, duration,e):
        self.text_browser.append("Performing emotion analysis...")
        frames = []
        start_time = time.time()

        # Load the custom model
        custom_model = load_model("3-emotion_recognition_model.h5")  # Replace with your model path
        emotion_labels = ["disgusted", "happy", "neutral","sad"]
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (640, 480))
            frames.append(frame)

        try:
            if frames:
                last_frame = frames[-1]

                # Preprocess for custom model
                custom_frame = cv2.resize(last_frame, (224, 224))  # Resize to match VGG16 input size
                custom_frame = custom_frame / 255.0  # Normalize pixel values
                custom_frame = np.expand_dims(custom_frame, axis=0)  # Expand dimensions for batch size

                # Custom model prediction
                custom_predictions = custom_model.predict(custom_frame)[0]
                emotion_labels = ["disgusted", "happy", "neutral","sad"]  # Define labels as per dataset
                custom_emotion = emotion_labels[np.argmax(custom_predictions)]
                custom_confidence = np.max(custom_predictions)

                # DeepFace prediction
                emotion_results = DeepFace.analyze(
                    last_frame,
                    actions=["emotion"],
                    enforce_detection=False
                )

                if isinstance(emotion_results, list):
                    emotion_results = emotion_results[0]
                deepface_emotion = emotion_results.get("dominant_emotion", "unknown") 
                deepface_confidence = emotion_results["emotion"].get(deepface_emotion, 0)/100

                # Combine predictions with weighting
                weights = {"custom": 0.7, "deepface": 0.3}
                combined_emotion = (
                    custom_emotion if custom_confidence * weights["custom"] > deepface_confidence* weights["deepface"] 
                    else deepface_emotion
                )

                self.text_browser.append(f"Custom Model Emotion: {custom_emotion} (Confidence: {custom_confidence:.2f})")
                self.text_browser.append(f"DeepFace Emotion: {deepface_emotion} (Confidence: {deepface_confidence:.2f})")
                self.text_browser.append(f"Combined Emotion: {combined_emotion}")

                # Take action based on combined emotion
                if combined_emotion in ["disgust", "disappointed", "angry", "sad", "fear"]:
                    self.text_browser.append("Disliked track. Skipping to next song.")
                    next_button = self.browser.find_element(By.XPATH, "//*[@id='left-controls']/div/tp-yt-paper-icon-button[5]")
                    next_button.click()
                else:
                    self.text_browser.append("Enjoying the track!")
                e=combined_emotion
        except Exception as e:
            e="Neutral"
            self.text_browser.append(f"Error during emotion analysis: {e}")

    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame from BGR (OpenCV) to RGB (for QImage)
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform emotion analysis on the frame in real-time
            try:
                emotion_results = DeepFace.analyze(
                    frame_rgb,
                    actions=["emotion"],
                    enforce_detection=False
                )

                # Handle the case when the result is a list
                if isinstance(emotion_results, list):
                    emotion_results = emotion_results[0]

                # Extract the dominant emotion
                dominant_emotion = emotion_results.get("dominant_emotion", "unknown")

                # Overlay the emotion text on the frame
                self.display_emotion_on_feed(frame_rgb, dominant_emotion)
            except Exception as e:
                self.text_browser.append(f"Error during emotion analysis: {e}")

            # Display the camera feed on the UI
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def display_emotion_on_feed(self, frame, emotion):
        # Draw the emotion on the frame (in RGB format)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Emotion: {emotion}"
        color = (0, 255, 0)  # Green color for the text
        cv2.putText(frame, text, (10, 30), font, 0.8, color, 2, cv2.LINE_AA)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        self.browser.quit()
        event.accept()


class NovaVoiceAssistant:
    def __init__(self, text_browser, browser):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.text_browser = text_browser
        self.browser = browser
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 160)
        self.tts_engine.setProperty('volume', 1.0)
        self.current_emotion = "neutral"
        self.current_song_url = None
        self.song_queue = []
        self.is_song_paused = False
        self.paused_song_url = None
        self.paused_song_time = 0

        self.song_data = self.load_song_data("songss.xlsx")
        self.trigger_phrases = ["nova", "hey nova", "hello nova"]

    def load_song_data(self, file_path):
        try:
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active
            song_data = {
                "happy": [cell.value for cell in sheet["B"] if cell.row > 2 and cell.value],
                "sad": [cell.value for cell in sheet["E"] if cell.row > 2 and cell.value],
                "neutral": [cell.value for cell in sheet["H"] if cell.row > 2 and cell.value],
                "frustrated": [cell.value for cell in sheet["K"] if cell.row > 2 and cell.value],
            }
            return song_data
        except Exception as e:
            self.text_browser.append(f"Error loading song data: {e}")
            return {}

    def speak(self, text):
        self.text_browser.append(f"Nova: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.text_browser.append("Listening... (Say 'Nova' or 'Hey Nova')")

            while True:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    command = self.recognizer.recognize_google(audio).lower()
                    self.text_browser.append(f"Command Received: {command}")
                    if any(phrase in command for phrase in self.trigger_phrases):
                        self.respond(command)
                    elif "play song" in command:
                        self.play_any_song()
                    elif "resume" in command:
                        self.resume_song()
                    elif "pause" in command:
                        self.pause_song()
                    elif "stop" in command:
                        self.stop_song()
                    elif "next" in command or "skip" in command:
                        self.play_next_song()
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    self.text_browser.append(f"Speech recognition error: {e}")

    def respond(self, command):
        self.speak("Hello! How can I assist you?")

    def play_any_song(self):
        try:
            # Get the current emotion
            e="neutral"
            threading.Thread(target=self.perform_emotion_analysis, args=(7,e)).start()
            self.text_browser.append(f"{e}")
            if e not in self.song_data:
                self.speak(f"You seem netural.")
            # Fetch songs for the current emotion
            songs = self.song_data[e]
            if not songs:
                self.speak("You seem netural.")

            # Randomly select a song
            self.current_song_url = random.choice(songs)
            self.play_url(self.current_song_url)
        except:
            pass
            
    def resume_song(self):
        try:
            playbtn = self.browser.find_element(By.XPATH, "//*[@id='play-pause-button']")
            playbtn.click()
        except Exception as inner_e:
            self.text_browser.append(f"Error interacting with the browser's play button: {inner_e}")


    def pause_song(self):
        try:
            playbtn = self.browser.find_element(By.XPATH, "//*[@id='play-pause-button']")
            playbtn.click()
        except Exception as inner_e:
            self.text_browser.append(f"Error interacting with the browser's play button: {inner_e}")


    def stop_song(self):
        try:
            self.browser.execute_script("""
                let video = document.querySelector('video');
                if (video) {
                    video.pause();
                    video.currentTime = 0;
                }
            """)
            self.is_song_paused = False
            self.paused_song_url = None
            self.paused_song_time = 0
            self.speak("The song has been stopped.")
        except Exception as e:
            self.text_browser.append(f"Error stopping song: {e}")

    def play_next_song(self):
        try:
            playbtn = self.browser.find_element(By.XPATH, "//*[@id='left-controls']/div/tp-yt-paper-icon-button[5]")
            playbtn.click()
        except Exception as e:
            self.text_browser.append(f"Error skipping to next song: {e}")

    def play_url(self, url, start_time=0):
        try:
            self.browser.get(url)
            time.sleep(3)
            play_button = self.browser.find_element(By.XPATH, "//button[contains(@aria-label, 'Play')]")
            play_button.click()
            if start_time > 0:
                self.browser.execute_script(f"document.querySelector('video').currentTime = {start_time};")
            self.is_song_paused = False
        except Exception as e:
            self.text_browser.append(f"Error playing URL: {e}")

    def shutdown(self):
        self.tts_engine.stop()
if __name__ == "__main__":
    app = QApplication(sys.argv)
      # Apply dark theme
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window = YouTubeMusicEmotionAnalyzer()
    main_window.show()
    sys.exit(app.exec_())	
