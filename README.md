
# 🎧 Moodsync – Emotion-Based Music Recommender 🎶

Moodsync is an intelligent, real-time **emotion detection and music recommendation system** that analyzes a user's mood through their webcam and voice, and syncs music playback from YouTube Music accordingly.

> 😄 😐 😢 😠 — Your mood drives your music.

---

## 🚀 Features

- 🧠 **Dual Emotion Detection**: Combines predictions from a fine-tuned VGG16 CNN model and DeepFace.
- 📷 **Real-time Webcam Analysis**: Detects facial expressions live using OpenCV and DeepFace.
- 🎙️ **Voice Assistant – Nova**: Triggered by "Hey Nova", supports commands like "Play song", "Pause", "Next", etc.
- 🎵 **Music Integration**: Automatically skips disliked songs and selects songs from an emotion-tagged playlist (from `songss.xlsx`).
- 💻 **GUI App**: Built with PyQt5 for a modern desktop experience with camera feed and logs.
- 🎬 **Dataset Collection Tool**: Captures and labels face images based on emotion-induced audio (for training).

---

## 🛠️ Tech Stack

| Component         | Technology                                  |
|------------------|---------------------------------------------|
| UI / App         | Python, PyQt5, OpenCV, DeepFace             |
| ML Model         | TensorFlow (VGG16 Fine-Tuned Model)         |
| Voice Assistant  | SpeechRecognition, pyttsx3                  |
| Music Control    | Selenium (YouTube Music Web Interface)      |
| Dataset Creation | Pygame, OpenCV                              |
| Dataset Format   | `emotion_dataset/`, `emotion_dataset_split/`|

---

## 📂 Project Structure

```
📁 Moodsync/
├── App.py                     # Main GUI application
├── Model-FineTuner.py         # Train and fine-tune VGG16 model
├── ModelTester.py             # Live testing of the trained model
├── UserDataSetModeller.py     # Script to generate training dataset via webcam & audio
├── 3-emotion_recognition_model.h5  # Trained emotion model (not included here)
├── songss.xlsx                # Emotion-based song links
├── *.mp3                      # Emotion-inducing audio files
```

---

## 🔧 Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/moodsync.git
cd moodsync
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

Recommended libraries:
- opencv-python
- PyQt5
- selenium
- tensorflow
- deepface
- pygame
- qdarkstyle
- speechrecognition
- pyttsx3
- webdriver-manager
- openpyxl

3. **Prepare WebDriver**
> Chrome is used via `webdriver-manager`. No manual driver setup is needed.

---

## ▶️ How to Run

### Run the GUI App:
```bash
python App.py
```

### Generate Dataset (with audio prompts):
```bash
python UserDataSetModeller.py
```

### Train Custom Emotion Model:
```bash
python Model-FineTuner.py
```

### Test Emotion Detection Model:
```bash
python ModelTester.py
```

---

## 📊 Emotion Classes

- `happy`
- `sad`
- `neutral`
- `disgusted`

These emotions correspond to the dataset folders and song categories in `songss.xlsx`.

---

## 📌 Notes

- Emotion detection combines both DeepFace and a custom-trained VGG16 model.
- The song recommendation system is rule-based using a pre-defined Excel sheet (`songss.xlsx`).
- Voice control requires a working microphone and quiet background.

---

## 🙌 Credits

Developed by Likhith Gujjar A, Manasa S N and Prince Jain.
Project inspired by emotion-aware computing and music therapy.

---

## 🪪 License

MIT License – free to use and modify.
