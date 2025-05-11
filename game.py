import os
import tkinter as tk
import random
from PIL import Image, ImageTk

class Game:
    def __init__(self, root):
        self.root = root
        self.cell_size = 50
        self.grid_size = 6
        self.canvas = None
        self.player_pos = [1, 1]
        self.player_img = None
        self.floor_img = None
        self.fruit_img = None
        self.canvas_person = None
        self.canvas_fruit = None
        self.score = 0
        self.score_label = None
        
        # Initialize UI elements
        self.setup_score_display()
        
    def setup_score_display(self):
        self.score_label = tk.Label(self.root, text=f"Score: {self.score}", font=("Arial", 14))
        self.score_label.pack()

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
            self.check_collision()

    def check_collision(self):
        # Get fruit position (in grid coordinates)
        fruit_coords = self.canvas.coords(self.canvas_fruit)
        fruit_grid_pos = [int(fruit_coords[0]/self.cell_size), int(fruit_coords[1]/self.cell_size)]
        
        if self.player_pos == fruit_grid_pos:
            self.score += 10
            self.score_label.config(text=f"Score: {self.score}")
            self.move_fruit()

    def move_fruit(self):
        # Remove old fruit
        self.canvas.delete(self.canvas_fruit)
        
        # Generate new random position (not same as player)
        while True:
            fruit_pos = [
                random.randint(0, self.grid_size-1),
                random.randint(0, self.grid_size-1)
            ]
            if fruit_pos != self.player_pos:
                break
                
        # Create new fruit
        self.canvas_fruit = self.canvas.create_image(
            fruit_pos[0] * self.cell_size,
            fruit_pos[1] * self.cell_size,
            image=self.fruit_img,
            anchor="nw"
        )

    def elements_init(self):
        # Load images
        try:
            self.player_img = ImageTk.PhotoImage(
                Image.open(os.path.join("pictures", "neutral.png")).resize((self.cell_size, self.cell_size)))
            self.floor_img = ImageTk.PhotoImage(
                Image.open(os.path.join("pictures", "floor.png")).resize((self.cell_size, self.cell_size))
            )
            self.fruit_img = ImageTk.PhotoImage(
                Image.open(os.path.join("pictures", "fruit.png")).resize((self.cell_size, self.cell_size))
            )
        except FileNotFoundError as e:
            print(f"Image not found: {e}")
            return

        # Create game grid
        grid_frame = tk.Frame(self.root)
        grid_frame.pack(pady=10)

        self.canvas = tk.Canvas(
            grid_frame,
            width=self.cell_size * self.grid_size,
            height=self.cell_size * self.grid_size,
            bg="white"
        )
        self.canvas.pack()

        # Draw background
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.canvas.create_image(
                    col * self.cell_size,
                    row * self.cell_size,
                    image=self.floor_img,
                    anchor="nw"
                )

        # Add player
        self.canvas_person = self.canvas.create_image(
            self.player_pos[0] * self.cell_size,
            self.player_pos[1] * self.cell_size,
            image=self.player_img,
            anchor="nw"
        )

        # Add initial fruit
        self.move_fruit()

        # Bind controls
        self.root.bind("<KeyPress>", self.move_person)
        self.root.focus_set()