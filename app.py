import tkinter as tk
import random
from camera2 import EmotionDetector
from game import Game


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion Game")
        self.game = Game(self)
        self.game.elements_init()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()