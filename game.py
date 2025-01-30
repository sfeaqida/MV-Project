import pygame
import sys
import cv2
import numpy as np
import random
from ultralytics import YOLO

# Initialize Pygame
pygame.init()

# Load YOLO Model
model = YOLO("D:/MV Class/best (2).pt")  # Ensure the path to your model is correct

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Visionary Vision")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 102, 255)
YELLOW = (255, 204, 0)
RED = (255, 0, 0)

# Initialize music
pygame.mixer.init()
music_playing = True
pygame.mixer.music.load("music.mp3")  # Replace with the actual path to the music file
pygame.mixer.music.play(-1)  # Loop the music indefinitely

# Load icons
music_icon = pygame.image.load("D:\MV Class\sound1.jpeg")  # Replace with the actual path to the music icon
music_icon = pygame.transform.scale(music_icon, (50, 50))
home_icon = pygame.image.load("D:\MV Class\home.jpeg")  # Replace with the actual path to the home icon
home_icon = pygame.transform.scale(home_icon, (50, 50))

# Utility function to display text
def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect(center=(x, y))
    surface.blit(text_obj, text_rect)

# Gesture recognition function using the YOLO model
def predict_gesture(frame):
    """
    Use the YOLO model to predict the gesture in the given frame.
    """
    results = model(frame)  # Run the model on the frame
    predictions = results[0]  # Get the first batch of results

    if len(predictions.boxes) > 0:
        # Assuming the gesture class with the highest confidence is desired
        top_prediction = max(predictions.boxes, key=lambda x: x.conf)
        cls_idx = int(top_prediction.cls)  # Get the predicted class index
        cls_name = model.names[cls_idx]  # Get the class name
        return cls_name
    else:
        return None  # No gesture detected


# Button class for clickable buttons
class IconButton:
    def __init__(self, x, y, image, action):
        self.image = image
        self.rect = self.image.get_rect(topleft=(x, y))
        self.action = action

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# Main menu
def main_menu():
    font = pygame.font.Font(None, 50)
    running = True

    start_button = Button(300, 250, 200, 50, "Start", BLUE, YELLOW, start_game)
    exit_button = Button(300, 350, 200, 50, "Exit", BLUE, YELLOW, sys.exit)
    music_button = Button(300, 450, 200, 50, "Sound 🔇", RED, YELLOW, toggle_music)


    while running:
        # Create gradient background
        for y in range(SCREEN_HEIGHT):
            color = [int(BLUE[i] + (YELLOW[i] - BLUE[i]) * y / SCREEN_HEIGHT) for i in range(3)]
            pygame.draw.line(screen, color, (0, y), (SCREEN_WIDTH, y))

        draw_text("Master the Sign: SignL. Challenge", font, WHITE, screen, SCREEN_WIDTH // 2, 100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if start_button.is_clicked(event):
                start_button.action()
            if exit_button.is_clicked(event):
                exit_button.action()
            if music_button.is_clicked(event):
                music_button.action()

        start_button.draw(screen, font)
        exit_button.draw(screen, font)
        music_button.draw(screen, font)

        pygame.display.flip()
        
# Button class for clickable buttons
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.action = action

    def draw(self, surface, font):
        mouse_pos = pygame.mouse.get_pos()
        color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=15)  # Rounded rectangle
        draw_text(self.text, font, BLACK, surface, self.rect.centerx, self.rect.centery)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# Toggle music function
def toggle_music():
    global music_playing
    if music_playing:
        pygame.mixer.music.pause()
    else:
        pygame.mixer.music.unpause()
    music_playing = not music_playing

# Start game screen
def start_game():
    font = pygame.font.Font(None, 50)
    running = True

    level1_button = Button(300, 250, 200, 50, "Level 1", BLUE, YELLOW, level_1)
    level2_button = Button(300, 350, 200, 50, "Level 2", BLUE, YELLOW, level_2)
    home_button = IconButton(20, 20, home_icon, main_menu)

    while running:
        screen.fill(WHITE)
        draw_text("Choose Level", font, BLACK, screen, SCREEN_WIDTH // 2, 100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if level1_button.is_clicked(event):
                level1_button.action()
            if level2_button.is_clicked(event):
                level2_button.action()
            if home_button.is_clicked(event):
                home_button.action()

        level1_button.draw(screen, font)
        level2_button.draw(screen, font)
        home_button.draw(screen)

        pygame.display.flip()

# Level 1: Alphabet recognition
def level_1():
    font = pygame.font.Font(None, 50)
    running = True

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Target gesture for the level
    target_gesture = random.choice(list(model.names.values()))  # Random target from model classes
    score = 0
    max_score = 5  # End level when score reaches this value

    home_button = IconButton(20, 20, home_icon, main_menu)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)

        # Predict gesture using YOLO model
        predicted_gesture = predict_gesture(frame)

        # Clear the screen
        screen.fill(WHITE)

        # Draw the video frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        screen.blit(frame_surface, (100, 150))  # Adjust position as needed

        # Check if the prediction matches the target
        if predicted_gesture == target_gesture:
            score += 1
            target_gesture = random.choice(list(model.names.values()))  # Change target

        # Display instructions and score
        draw_text(f"Perform Gesture: {target_gesture}", font, BLACK, screen, SCREEN_WIDTH // 2, 50)
        draw_text(f"Score: {score}/{max_score}", font, WHITE, screen, SCREEN_WIDTH // 2, 500)

        # End level if max score is reached
        if score >= max_score:
            cap.release()
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if home_button.is_clicked(event):
                cap.release()
                home_button.action()

        home_button.draw(screen)
        pygame.display.flip()

# Level 2: Spell random words
def level_2():
    font = pygame.font.Font(None, 50)
    running = True

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # List of short random words
    words = ["ayam", "farah", "ikan", "kaca", "beg"]
    target_word = random.choice(words)  # Random target word
    current_index = 0

    home_button = IconButton(20, 20, home_icon, main_menu)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)

        # Predict gesture using YOLO model
        predicted_gesture = predict_gesture(frame)

        # Clear the screen
        screen.fill(WHITE)

        # Draw the video frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        screen.blit(frame_surface, (100, 150))  # Adjust position as needed

        # Check if the prediction matches the current letter in the target word
        if predicted_gesture == target_word[current_index].upper():
            current_index += 1

            # Check if the entire word is completed
            if current_index >= len(target_word):
                # Display "Amazing!" before moving to the next word
                draw_text("Amazing!", font, YELLOW, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                pygame.display.flip()
                pygame.time.delay(1000)  # Pause for 1 second

                target_word = random.choice(words)  # Change target word
                current_index = 0

        # Display instructions and progress
        draw_text(f"Spell the Word: {target_word}", font, BLACK, screen, SCREEN_WIDTH // 2, 50)
        draw_text(f"Current Letter: {target_word[current_index].upper()}", font, BLACK, screen, SCREEN_WIDTH // 2, 100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if home_button.is_clicked(event):
                cap.release()
                home_button.action()

        home_button.draw(screen)
        pygame.display.flip()

# Main game function
def main_game():
    main_menu()

# Run the game
if __name__ == "__main__":
    main_game()
