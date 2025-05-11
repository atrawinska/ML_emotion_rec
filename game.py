import os
import tkinter as tk
from PIL import Image, ImageTk

class Game:
    def __init__(self, root):
        self.root = root
        self.cell_size = 50
        self.grid_size = 6
        self.canvas = None
        self.player_pos = [1, 1]  # Using [col, row] format
        self.player_img = None
        self.floor_img = None
        self.canvas_person = None

    def move_person(self, event):
        new_pos = self.player_pos.copy()
        
        if event.keysym == "Up":
            new_pos[1] = max(0, self.player_pos[1] - 1)
        elif event.keysym == "Down":
            new_pos[1] = min(self.grid_size-1, self.player_pos[1] + 1)
        elif event.keysym == "Left":
            new_pos[0] = max(0, self.player_pos[0] - 1)
        elif event.keysym == "Right":
            new_pos[0] = min(self.grid_size-1, self.player_pos[0] + 1)
        
        if new_pos != self.player_pos:
            self.player_pos = new_pos
            self.canvas.coords(
                self.canvas_person,
                self.player_pos[0] * self.cell_size,
                self.player_pos[1] * self.cell_size
            )

    def elements_init(self):
        # Load and resize images
        try:
            self.player_img = ImageTk.PhotoImage(
                Image.open(os.path.join("pictures", "neutral.png")).resize((self.cell_size, self.cell_size))
            )
            self.floor_img = ImageTk.PhotoImage(
                Image.open(os.path.join("pictures", "floor.png")).resize((self.cell_size, self.cell_size))
            )
        except FileNotFoundError as e:
            print(f"Image not found: {e}")
            return

        # Create a frame for the grid
        grid_frame = tk.Frame(self.root)
        grid_frame.pack(pady=10)

        self.canvas = tk.Canvas(
            grid_frame,
            width=self.cell_size * self.grid_size,
            height=self.cell_size * self.grid_size,
            bg="white"
        )
        self.canvas.pack()

        # Draw the background grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.canvas.create_image(
                    col * self.cell_size,
                    row * self.cell_size,
                    image=self.floor_img,
                    anchor="nw"
                )

        # Add player at starting position
        self.canvas_person = self.canvas.create_image(
            self.player_pos[0] * self.cell_size,
            self.player_pos[1] * self.cell_size,
            image=self.player_img,
            anchor="nw"
        )

        # Bind arrow keys
        self.root.bind("<KeyPress>", self.move_person)
        self.root.focus_set()  # Ensure window has focus for key events