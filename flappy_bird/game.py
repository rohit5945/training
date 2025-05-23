import pygame
import sys
import os

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255) # For background, or can use an image

# Game constants
FPS = 60
ASSETS_DIR = "assets" # Corrected path

import pygame
import sys
import os
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255) # For background, or can use an image
GREEN = (0, 255, 0) # For pipes

# Game constants
FPS = 60
ASSETS_DIR = "assets" 

# Pipe Constants
PIPE_WIDTH = 70
PIPE_GAP = 150  # Gap between upper and lower pipes
PIPE_SPEED = 3
PIPE_SPAWN_RATE = 1500 # milliseconds (1.5 seconds)


# Bird Class
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.frames = []
        self.frames.append(pygame.image.load(os.path.join(ASSETS_DIR, "bird_wing_up.png")).convert_alpha())
        self.frames.append(pygame.image.load(os.path.join(ASSETS_DIR, "bird_wing_down.png")).convert_alpha())
        
        self.current_frame = 0
        self.image = self.frames[self.current_frame]
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
        
        self.animation_timer = 0
        self.animation_delay = 10 

        self.velocity = 0
        self.gravity = 0.5
        self.flap_strength = -9

    def update(self):
        # Animation
        self.animation_timer += 1
        if self.animation_timer >= self.animation_delay:
            self.animation_timer = 0
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.image = self.frames[self.current_frame]
        
        # Physics
        self.velocity += self.gravity
        self.rect.y += self.velocity

        if self.rect.top < 0:
            self.rect.top = 0
            self.velocity = 0

    def flap(self):
        self.velocity = self.flap_strength

    # No separate draw method needed if part of all_sprites and drawn via group.draw()

# Pipe Class
class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, is_top_pipe):
        super().__init__()
        self.image = pygame.Surface((PIPE_WIDTH, SCREEN_HEIGHT // 2)) # Placeholder height
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()

        if is_top_pipe:
            self.image = pygame.transform.flip(self.image, False, True) # Flip if top pipe
            self.rect.bottomleft = (x, y - PIPE_GAP // 2)
        else:
            self.rect.topleft = (x, y + PIPE_GAP // 2)
            
        self.passed = False # For scoring

    def update(self):
        self.rect.x -= PIPE_SPEED
        if self.rect.right < 0:
            self.kill() # Remove pipe if it's off-screen


# Game variables
clock = pygame.time.Clock()
bird = Bird()
all_sprites = pygame.sprite.Group()
pipes_group = pygame.sprite.Group()
all_sprites.add(bird)

game_active = True
score = 0
font = pygame.font.SysFont(None, 55) # For score display

# Timer for pipe spawning
SPAWNPIPE = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWNPIPE, PIPE_SPAWN_RATE)

def create_pipe_pair():
    # Randomize the y position of the gap center
    gap_center_y = random.randint(PIPE_GAP // 2 + 50, SCREEN_HEIGHT - PIPE_GAP // 2 - 50)
    
    top_pipe = Pipe(SCREEN_WIDTH, gap_center_y, True)
    bottom_pipe = Pipe(SCREEN_WIDTH, gap_center_y, False)
    
    pipes_group.add(top_pipe, bottom_pipe)
    all_sprites.add(top_pipe, bottom_pipe)

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_active:
                bird.flap()
            if event.key == pygame.K_SPACE and not game_active: # Restart game
                # Reset game state (to be fully implemented later)
                bird.rect.center = (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
                bird.velocity = 0
                pipes_group.empty()
                all_sprites.empty()
                all_sprites.add(bird)
                score = 0
                game_active = True
        
        if event.type == SPAWNPIPE and game_active:
            create_pipe_pair()

    # --- Game logic updates ---
    if game_active:
        all_sprites.update() # Calls update() on bird and all pipes

        # Collision detection
        # Bird hits pipes
        if pygame.sprite.spritecollide(bird, pipes_group, False):
            print("Game Over - Collided with Pipe")
            game_active = False
        
        # Bird hits bottom of screen
        if bird.rect.bottom >= SCREEN_HEIGHT:
            print("Game Over - Hit Ground")
            bird.rect.bottom = SCREEN_HEIGHT # Keep bird on ground
            bird.velocity = 0 # Stop falling
            game_active = False
        
        # Bird hits top of screen (already handled in Bird.update to prevent going off, but can also be a game over)
        # if bird.rect.top <= 0:
        #     print("Game Over - Hit Ceiling") 
        #     game_active = False


        # Scoring logic
        for pipe in pipes_group:
            if not pipe.passed and bird.rect.left > pipe.rect.right:
                # Bird has passed the pipe's right edge
                # Increment score by 0.5 for each pipe in a pair. Total 1 point per pair.
                score += 0.5
                pipe.passed = True # Mark this specific pipe as passed
                print(f"Score updated: {int(score)}")
    
    # --- Drawing code ---
    SCREEN.fill(BLUE)
    all_sprites.draw(SCREEN) # Draws bird and all pipes
    
    # Draw score
    score_text = font.render(f"Score: {int(score)}", True, WHITE) # Display integer score
    SCREEN.blit(score_text, (10, 10))

    if not game_active:
        game_over_text = font.render("Game Over!", True, WHITE)
        restart_text = font.render("Press SPACE to Restart", True, WHITE)
        SCREEN.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 3))
        SCREEN.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2))


    pygame.display.flip()
    clock.tick(FPS)        # Cap the frame rate

# Quit Pygame
pygame.quit()
sys.exit()
