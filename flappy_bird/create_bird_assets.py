import pygame
import os

# Initialize Pygame (needed for drawing and image saving)
pygame.init()

# Ensure the assets directory exists
assets_dir = "flappy_bird/assets" # Corrected path
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

# Image settings
BIRD_WIDTH = 34  # Slightly wider for wings
BIRD_HEIGHT = 24 # Standard height
BODY_COLOR = (255, 255, 0)  # Yellow
WING_COLOR = (200, 200, 0)  # Darker Yellow
BEAK_COLOR = (255, 165, 0) # Orange
EYE_COLOR = (0, 0, 0)      # Black

# --- Create bird_wing_up.png ---
image_up = pygame.Surface((BIRD_WIDTH, BIRD_HEIGHT), pygame.SRCALPHA) # SRCALPHA for transparency
image_up.fill((0,0,0,0)) # Transparent background

# Body
pygame.draw.ellipse(image_up, BODY_COLOR, (BIRD_WIDTH // 2 - 10, BIRD_HEIGHT // 2 - 7, 20, 14))
# Wing up
pygame.draw.ellipse(image_up, WING_COLOR, (BIRD_WIDTH // 2 - 8, BIRD_HEIGHT // 2 - 12, 16, 10)) # Wing slightly higher
# Beak
pygame.draw.polygon(image_up, BEAK_COLOR, [(BIRD_WIDTH - 5, BIRD_HEIGHT // 2), (BIRD_WIDTH - 12, BIRD_HEIGHT // 2 - 3), (BIRD_WIDTH - 12, BIRD_HEIGHT // 2 + 3)])
# Eye
pygame.draw.circle(image_up, EYE_COLOR, (BIRD_WIDTH - 10, BIRD_HEIGHT // 2 - 2), 2)

pygame.image.save(image_up, os.path.join(assets_dir, "bird_wing_up.png"))
print(f"Saved {os.path.join(assets_dir, 'bird_wing_up.png')}")

# --- Create bird_wing_down.png ---
image_down = pygame.Surface((BIRD_WIDTH, BIRD_HEIGHT), pygame.SRCALPHA)
image_down.fill((0,0,0,0)) # Transparent background

# Body
pygame.draw.ellipse(image_down, BODY_COLOR, (BIRD_WIDTH // 2 - 10, BIRD_HEIGHT // 2 - 7, 20, 14))
# Wing down
pygame.draw.ellipse(image_down, WING_COLOR, (BIRD_WIDTH // 2 - 8, BIRD_HEIGHT // 2 - 2, 16, 10)) # Wing slightly lower
# Beak
pygame.draw.polygon(image_down, BEAK_COLOR, [(BIRD_WIDTH - 5, BIRD_HEIGHT // 2), (BIRD_WIDTH - 12, BIRD_HEIGHT // 2 - 3), (BIRD_WIDTH - 12, BIRD_HEIGHT // 2 + 3)])
# Eye
pygame.draw.circle(image_down, EYE_COLOR, (BIRD_WIDTH - 10, BIRD_HEIGHT // 2 - 2), 2)

pygame.image.save(image_down, os.path.join(assets_dir, "bird_wing_down.png"))
print(f"Saved {os.path.join(assets_dir, 'bird_wing_down.png')}")

pygame.quit()
