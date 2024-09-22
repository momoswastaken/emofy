# 🎵 EMOFY: Emotion-based Music Recommendation System

**EMOFY** is an innovative music recommendation system that analyzes facial expressions and suggests music tracks tailored to the user's mood. By leveraging cutting-edge technologies like computer vision, machine learning, and facial recognition, this app provides a personalized musical experience based on real-time emotional analysis.

## 🌟 Features
- 🎶 **Emotion Detection:** Detects emotions such as happiness, sadness, anger, and more using **facial recognition** powered by **OpenCV** and **Keras**.
- 🤖 **Machine Learning Integration:** Uses **TensorFlow** and **Keras** for emotion classification through deep learning models.
- 🎧 **Real-time Recommendations:** Offers instant music recommendations that align with your emotional state.
- 💻 **User-friendly Interface:** Built using **Streamlit**, making it interactive and easy to navigate.

## 🛠️ Technologies
- **Python**: Core programming language for building the system.
- **OpenCV**: For real-time facial detection and image processing.
- **TensorFlow & Keras**: Deep learning frameworks for emotion classification.
- **Machine Learning**: For training emotion detection models.
- **Facial Recognition**: To accurately capture and analyze user emotions.
- **Streamlit**: For building a fast, responsive, and interactive web app interface.

## 📚 Prerequisites
Before you can run EMOFY, ensure you have the following installed:
- Python 3.x
- OpenCV
- TensorFlow
- Keras
- Streamlit

You can install the required libraries using:
```bash
pip install -r requirements.txt
```

## 🚀 Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/momoswastaken/emofy.git
   cd emofy
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Allow access to your camera for emotion detection, and enjoy personalized music recommendations!

## 🤔 How it Works
- **Facial Detection:** EMOFY captures a live video stream using your webcam.
- **Emotion Analysis:** The system analyzes your facial expressions in real-time, predicting emotions such as happiness, sadness, and surprise.
- **Music Recommendation:** Based on the detected emotion, EMOFY curates a playlist from a predefined database of songs that best fit your mood.

## 🤩 Future Enhancements
- Integration with **Spotify API** for dynamic, real-time playlist recommendations.
- Enhancing emotion detection accuracy with more diverse training data.
- Adding more emotion categories to better reflect complex moods.

## 🏆 Contribution
We welcome contributions! Feel free to open issues, discuss ideas, or submit pull requests.

