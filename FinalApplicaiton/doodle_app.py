import pygame
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- CONFIGURATION ---
# Window settings
WIDTH, HEIGHT = 700, 520
CANVAS_WIDTH, CANVAS_HEIGHT = 400, 400
BRUSH_SIZE = 16 # The diameter of the drawing brush

# Colors for the new dark theme
BACKGROUND = (40, 42, 54)
CANVAS_BG = (20, 21, 27)
WHITE = (248, 248, 242)
LIGHT_GRAY = (98, 114, 164)
GREEN = (80, 250, 123)
RED = (255, 85, 85)
ORANGE = (255, 184, 108)
CYAN = (139, 233, 253)

# Model and class names
MODEL_PATH = "QuickDraw_CNN_35_classes.h5"
CLASS_NAMES = [
    "apple", "arm", "banana", "baseball bat", "beard", "bed", "bicycle",
    "book", "bowtie", "bridge", "camera", "car", "cat", "chair", "circle",
    "cloud", "computer", "crown", "cup", "dog", "door", "eye", "face",
    "flower", "fork", "guitar", "hamburger", "house", "ice cream", "key",
    "laptop", "leaf", "light bulb", "lightning", "line"
]

# --- SETUP ---
# Check if the model file exists before starting
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please make sure the H5 file is in the same directory as this script.")
    exit()

# Initialize Pygame
pygame.init()
pygame.font.init()

# Create the display window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Doodle Guesser AI")

# Create a separate surface for the drawing canvas
canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
canvas.fill(CANVAS_BG)

# Fonts
title_font = pygame.font.SysFont("Segoe UI", 36, bold=True)
main_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
prediction_font = pygame.font.SysFont("Segoe UI", 20)

# --- LOAD THE MODEL ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    status_text = "Model loaded. Draw something!"
except Exception as e:
    print(f"Error loading model: {e}")
    status_text = "Error: Could not load model."
    model = None

# --- HELPER FUNCTIONS ---
def preprocess_image(surface):
    """
    Takes a Pygame surface, converts it to a PIL image,
    and preprocesses it to be ready for the model.
    """
    image_data = pygame.image.tostring(surface, "RGB")
    pil_image = Image.frombytes("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), image_data)
    
    # Convert to grayscale and resize
    processed_image = pil_image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    
    # The canvas is black with white ink, so we don't need to invert.
    
    # Convert to a numpy array, normalize, and reshape
    np_image = np.array(processed_image).astype("float32") / 255.0
    np_image = np_image.reshape(1, 28, 28, 1)
    
    return np_image

def draw_button(surface, rect, text, bg_color, hover_color, is_hovered):
    """Draws a button with a hover effect."""
    color = hover_color if is_hovered else bg_color
    pygame.draw.rect(surface, color, rect, border_radius=10)
    
    text_surf = main_font.render(text, True, BACKGROUND)
    text_rect = text_surf.get_rect(center=rect.center)
    surface.blit(text_surf, text_rect)

def draw_screen(canvas_rect, guess_button_rect, clear_button_rect, is_guess_hovered, is_clear_hovered, predictions, thinking):
    """Handles all the drawing for the main screen."""
    screen.fill(BACKGROUND)

    # Draw title
    title_surf = title_font.render("Doodle Guesser AI", True, WHITE)
    screen.blit(title_surf, (20, 20))
    
    # Draw canvas
    screen.blit(canvas, canvas_rect)
    pygame.draw.rect(screen, LIGHT_GRAY, canvas_rect, 2, border_radius=5)
    
    # Draw UI panel on the right
    ui_panel_rect = pygame.Rect(CANVAS_WIDTH + 40, 0, WIDTH - CANVAS_WIDTH - 40, HEIGHT)
    
    # Draw buttons
    draw_button(screen, guess_button_rect, "Guess", GREEN, (110, 255, 178), is_guess_hovered)
    draw_button(screen, clear_button_rect, "Clear", RED, (255, 121, 121), is_clear_hovered)

    # Draw predictions header
    pred_header_surf = main_font.render("Top Predictions:", True, CYAN)
    screen.blit(pred_header_surf, (ui_panel_rect.left + 20, 240))
    
    # Display "Thinking..." message or the predictions
    if thinking:
        thinking_surf = prediction_font.render("Thinking...", True, ORANGE)
        screen.blit(thinking_surf, (ui_panel_rect.left + 20, 280))
    elif predictions:
        for i, (name, conf) in enumerate(predictions):
            text = f"{i+1}. {name.title()}"
            conf_text = f"{conf:.1f}%"
            
            text_surf = prediction_font.render(text, True, WHITE)
            conf_surf = prediction_font.render(conf_text, True, LIGHT_GRAY)
            
            screen.blit(text_surf, (ui_panel_rect.left + 20, 280 + i * 35))
            screen.blit(conf_surf, (ui_panel_rect.right - conf_surf.get_width() - 20, 280 + i * 35))

    pygame.display.flip()

def main():
    """Main application loop."""
    running = True
    drawing = False
    thinking = False
    last_pos = None # <-- FIX: To store the last mouse position for smooth lines
    predictions = []

    canvas_rect = canvas.get_rect(topleft=(20, 80))
    guess_button_rect = pygame.Rect(CANVAS_WIDTH + 60, 100, 180, 50)
    clear_button_rect = pygame.Rect(CANVAS_WIDTH + 60, 170, 180, 50)

    is_guess_hovered = False
    is_clear_hovered = False

    clock = pygame.time.Clock()

    while running:
        mouse_pos = pygame.mouse.get_pos()
        is_guess_hovered = guess_button_rect.collidepoint(mouse_pos)
        is_clear_hovered = clear_button_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if canvas_rect.collidepoint(mouse_pos):
                    drawing = True
                    # <-- FIX: Set the initial drawing position
                    last_pos = (mouse_pos[0] - canvas_rect.left, mouse_pos[1] - canvas_rect.top)
                elif is_guess_hovered and not thinking:
                    if model:
                        thinking = True
                        predictions = []
                elif is_clear_hovered:
                    canvas.fill(CANVAS_BG)
                    predictions = []
            
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_pos = None # <-- FIX: Reset last position when mouse is up
            
            if event.type == pygame.MOUSEMOTION and drawing:
                # <-- FIX: Draw a line from the last point to the current point
                current_pos = (mouse_pos[0] - canvas_rect.left, mouse_pos[1] - canvas_rect.top)
                if last_pos:
                    # Draw a line and circles at both ends for a smooth, thick stroke
                    pygame.draw.line(canvas, WHITE, last_pos, current_pos, BRUSH_SIZE)
                    pygame.draw.circle(canvas, WHITE, last_pos, BRUSH_SIZE // 2)
                    pygame.draw.circle(canvas, WHITE, current_pos, BRUSH_SIZE // 2)
                last_pos = current_pos # Update the last position

        # Handle prediction logic outside the event loop
        if thinking:
            draw_screen(canvas_rect, guess_button_rect, clear_button_rect, is_guess_hovered, is_clear_hovered, predictions, thinking)
            
            processed_img = preprocess_image(canvas)
            prediction_probs = model.predict(processed_img)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(prediction_probs)[-3:][::-1]
            predictions = [(CLASS_NAMES[i], prediction_probs[i] * 100) for i in top_indices]
            
            thinking = False # Prediction is done

        # Draw all UI elements
        draw_screen(canvas_rect, guess_button_rect, clear_button_rect, is_guess_hovered, is_clear_hovered, predictions, thinking)
        clock.tick(120) # Increased FPS for smoother drawing

    pygame.quit()

if __name__ == "__main__":
    main()
