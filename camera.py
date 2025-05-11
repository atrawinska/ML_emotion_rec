import tkinter as tk
from tkinter import Toplevel
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
import random

# Load your trained model
model = load_model("model.h5")

# Emotion labels in order
emotion_labels = ["neutral", "happy", "sad", "surprised", "mad", "disgusted", "fearful"]

class EmotionDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.current_emotion = "neutral"
        
    def get_emotion(self):
        return self.current_emotion
    
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            self.current_emotion = "neutral"  # Default emotion

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = model.predict(roi, verbose=0)[0]
                self.current_emotion = emotion_labels[np.argmax(preds)]

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, self.current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Convert frame to RGB and return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return rgb
        return None

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Expression Recognition")
        self.root.iconbitmap("logo.ico")
        self.root.configure(bg="#ffffff")
        
        # Create game window
        self.game_window = Toplevel(root)
        self.game_window.title("Emotion Control Game")
        self.game_window.configure(bg="#ffffff")
        
        # Initialize emotion detector
        self.emotion_detector = EmotionDetector()
        
        # Setup camera display
        self.camera_label = tk.Label(self.root, bg="#dcdcdc")
        self.camera_label.pack(side="left", padx=20, pady=20)
        
        # Setup girl image display
        self.girl_label = tk.Label(self.root, bg="#f6f5f3")
        self.girl_label.pack(side="right", padx=20, pady=20)
        
        # Initialize the game
        self.game = Game(self.game_window, self.emotion_detector)
        
        # Start updates
        self.update()
        
    def update(self):
        # Update camera feed
        frame = self.emotion_detector.update()
        if frame is not None:
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.config(image=imgtk)
            
            # Update girl image based on current emotion
            try:
                emotion = self.emotion_detector.get_emotion()
                img = Image.open(os.path.join("pictures", f"{emotion}.png")).resize((250, 250))
                self.girl_img = ImageTk.PhotoImage(img)
                self.girl_label.config(image=self.girl_img)
            except:
                pass
        
        self.root.after(10, self.update)

class Game:
    def __init__(self, root, emotion_detector):
        self.root = root
        self.emotion_detector = emotion_detector
        self.cell_size = 50
        self.grid_size = 5
        self.canvas = None
        self.player_pos = [1, 1]
        self.player_img = None
        self.floor_img = None
        self.fruit_img = None
        self.canvas_person = None
        self.canvas_fruit = None
        self.score = 0
        self.score_label = None
        self.current_emotion = "neutral"
        
        # Initialize UI elements
        self.setup_score_display()
        self.elements_init()
        
        # Start emotion checking
        self.setup_emotion_check()
        
    def setup_score_display(self):
        self.score_label = tk.Label(self.root, text=f"Score: {self.score}", font=("Arial", 14))
        self.score_label.pack()

    def setup_emotion_check(self):
        def check_emotion():
            if self.canvas:
                self.current_emotion = self.emotion_detector.get_emotion()
                self.move_person()
            self.root.after(500, check_emotion)
            
        check_emotion()

    def move_person(self, event=None):
        if not self.canvas:
            return
            
        new_pos = self.player_pos.copy()
        
        if self.current_emotion == "sad": #up
            new_pos[1] = max(0, self.player_pos[1] - 1)
        elif self.current_emotion == "happy": #down
            new_pos[1] = min(self.grid_size-1, self.player_pos[1] + 1)
        elif self.current_emotion == "mad": #left
            new_pos[0] = max(0, self.player_pos[0] - 1)
        elif self.current_emotion == "surprised": #right
            new_pos[0] = min(self.grid_size-1, self.player_pos[0] + 1)
        
        if new_pos != self.player_pos:
            self.player_pos = new_pos
            self.canvas.coords(
                self.canvas_person,
                self.player_pos[0] * self.cell_size,
                self.player_pos[1] * self.cell_size
            )
            self.check_collision()

    def check_collision(self):
        fruit_coords = self.canvas.coords(self.canvas_fruit)
        fruit_grid_pos = [int(fruit_coords[0]/self.cell_size), int(fruit_coords[1]/self.cell_size)]
        
        if self.player_pos == fruit_grid_pos:
            self.score += 10
            self.score_label.config(text=f"Score: {self.score}")
            self.move_fruit()

    def move_fruit(self):
        self.canvas.delete(self.canvas_fruit)
        
        while True:
            fruit_pos = [
                random.randint(0, self.grid_size-1),
                random.randint(0, self.grid_size-1)
            ]
            if fruit_pos != self.player_pos:
                break
                
        self.canvas_fruit = self.canvas.create_image(
            fruit_pos[0] * self.cell_size,
            fruit_pos[1] * self.cell_size,
            image=self.fruit_img,
            anchor="nw"
        )

    def elements_init(self):
        try:
            self.player_img = ImageTk.PhotoImage(
                Image.open(os.path.join("pictures", "neutral.png")).resize((self.cell_size, self.cell_size)))
            self.floor_img = ImageTk.PhotoImage(
                Image.open(os.path.join("pictures", "floor.png")).resize((self.cell_size, self.cell_size)))
            self.fruit_img = ImageTk.PhotoImage(
                Image.open(os.path.join("pictures", "fruit.png")).resize((self.cell_size, self.cell_size)))
        except FileNotFoundError as e:
            print(f"Image not found: {e}")
            return

        grid_frame = tk.Frame(self.root)
        grid_frame.pack()

        self.canvas = tk.Canvas(
            grid_frame,
            width=self.cell_size * self.grid_size,
            height=self.cell_size * self.grid_size,
            bg="white"
        )
        self.canvas.pack()

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.canvas.create_image(
                    col * self.cell_size,
                    row * self.cell_size,
                    image=self.floor_img,
                    anchor="nw"
                )

        self.canvas_person = self.canvas.create_image(
            self.player_pos[0] * self.cell_size,
            self.player_pos[1] * self.cell_size,
            image=self.player_img,
            anchor="nw"
        )

        self.move_fruit()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()