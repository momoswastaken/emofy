import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import queue
from os import listdir
from os.path import isfile, join
import time
from collections import Counter
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
)
capture_duration = 20
start_time = time.time()
emo = []
model = load_model('model1.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Neutral", 3: "Happy", 4: "Fearful", 5: "Sad", 6: "Surprised"}

def emotion_find():
    
    webrtc_ctx = webrtc_streamer(
            key="loopback",
            mode=WebRtcMode.SENDONLY,
            # client_settings=WEBRTC_CLIENT_SETTINGS,
        )
    st.markdown("## Click here to activate me")
    if(st.button("Activate EMP")):
        progress = st.progress(0)
        i=0
        while ( int(time.time() - start_time) < capture_duration and i<100):
            progress.progress(i+1)
            i=i+1
                # Find haar cascade to draw bounding box around face
            if webrtc_ctx.video_receiver:
                try:
                    video_frame = webrtc_ctx.video_receiver.get_frame(timeout=10)
                    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(video_frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
                    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

                    for (x, y, w, h) in faces:
                            #cv2.rectangle(video_frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                        roi_gray = gray[y:y + h, x:x + w]
                        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                        prediction = model.predict(cropped_img)
                        maxindex = int(np.argmax(prediction))
                        emo.append(emotion_dict[maxindex])
                except queue.Empty:
                    time.sleep(0.1)
                    continue
        if not emo:
            st.markdown("## Face Not Detected. Try Again")
        else:
            def most_frequent(List):
                occurence_count = Counter(List)
                return occurence_count.most_common(1)[0][0]
            user_emotion = most_frequent(emo)
            st.markdown("## You are "+user_emotion)
            songs = [f for f in listdir("songs/"+user_emotion) if isfile(join("songs/"+user_emotion, f))]
            for song in songs:
                st.markdown(song)
                st.audio("songs/"+user_emotion+"/"+song)

st.title("EmoFy")

nav = st.sidebar.radio("", ["Home","Play EmoFy"])

if nav == "Home":
    st.markdown("""<br>""", True)
    st.markdown(""" Welcome to EmoFy
Say goodbye to the same old playlists and hello to precision in emotion!     
Our cutting-edge tech decodes your mood with superhero-like accuracy, delivering a playlist 
that mirrors your emotions flawlessly. It's not just music; it's a mood-lifting adventure!
              <br /><br />
              Create your own musical masterpiece with EmoFy! Mix and match emotions 
                to craft custom playlists that scream YOU. 
                Share your musical mood with friends, and let the world 
                see the colorful symphony of your emotions. 
                It's time to express yourself through music like never before!
              <br />
             """, True)
    st.image("images/image-emo.jpg")

if nav == "Play EmoFy":
    emotion_find()
    