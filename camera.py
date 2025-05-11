import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

# Load your trained model
model = load_model("model.h5")

# Emotion labels in order
emotion_labels = ["neutral", "happy", "sad", "surprised", "mad", "disgusted", "fearful"]

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face expression recognition App")
        self.root.iconbitmap("logo.ico")
        self.root.configure(bg="#ffffff")
        self.root.geometry("1000x800")

        self.camera_label = tk.Label(self.root, bg="#dcdcdc")
        self.camera_label.pack(side="left", padx=20, pady=20)

        self.girl_label = tk.Label(self.root, bg="#f6f5f3")
        self.girl_label.pack(side="right", padx=20, pady=20)

        self.cap = cv2.VideoCapture(0)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            emotion = "neutral"

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = model.predict(roi, verbose=0)[0]
                emotion = emotion_labels[np.argmax(preds)]

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Update girl image
            try:
                img = Image.open(os.path.join("pictures", f"{emotion}.png")).resize((250, 250))
                self.girl_img = ImageTk.PhotoImage(img)
                self.girl_label.config(image=self.girl_img)
            except:
                pass

            # Convert frame to RGB and update camera label
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.config(image=imgtk)

        self.root.after(10, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
