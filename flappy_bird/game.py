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
PIPE_WIDTH = 70  # Consider scaling based on screen width, e.g., int(SCREEN_WIDTH * 0.0875)
PIPE_GAP = 150  # Gap between upper and lower pipes. Consider scaling, e.g., int(SCREEN_HEIGHT * 0.25)
PIPE_SPEED = 3  # Game speed. May need adjustment based on perceived speed on different devices/resolutions.
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

# Pipe Class
class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, is_top_pipe):
        super().__init__()
        # Create a surface for the pipe. Adjust height based on whether it's top or bottom.
        # For simplicity, we make it Screen height / 2, actual visible part is determined by position.
        self.image = pygame.Surface((PIPE_WIDTH, SCREEN_HEIGHT // 2 + PIPE_GAP)) # Generous height
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()

        if is_top_pipe:
            # Position the top pipe so its bottom edge is at 'y - PIPE_GAP // 2'
            self.rect.bottomleft = (x, y - PIPE_GAP // 2)
        else:
            # Position the bottom pipe so its top edge is at 'y + PIPE_GAP // 2'
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
# Consider dynamic font size based on screen height, e.g., int(SCREEN_HEIGHT * 0.09)
font = pygame.font.SysFont(None, 55) 

# Timer for pipe spawning
SPAWNPIPE = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWNPIPE, PIPE_SPAWN_RATE)

def create_pipe_pair():
    # Randomize the y position of the gap center
    # Ensure gap is not too close to screen top/bottom
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
            if event.key == pygame.K_SPACE:
                if game_active:
                    bird.flap()
                else: # Restart game on Space press if game over
                    bird.rect.center = (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
                    bird.velocity = 0
                    pipes_group.empty()
                    all_sprites.empty()
                    all_sprites.add(bird)
                    score = 0
                    game_active = True
        
        if event.type == pygame.MOUSEBUTTONDOWN: # Handle touch/mouse input
            if game_active:
                bird.flap()
            else: # Restart game on tap/click if game over
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
        all_sprites.update() 

        # Collision detection
        if pygame.sprite.spritecollide(bird, pipes_group, False):
            print("Game Over - Collided with Pipe")
            game_active = False
        
        if bird.rect.bottom >= SCREEN_HEIGHT:
            print("Game Over - Hit Ground")
            bird.rect.bottom = SCREEN_HEIGHT 
            bird.velocity = 0 
            game_active = False
        
        # Scoring logic
        for pipe in pipes_group:
            if not pipe.passed and bird.rect.left > pipe.rect.right:
                score += 0.5 
                pipe.passed = True 
                print(f"Score updated: {int(score)}")
    
    # --- Drawing code ---
    SCREEN.fill(BLUE)
    all_sprites.draw(SCREEN) 
    
    # Draw score
    # Consider relative score position, e.g., (SCREEN_WIDTH * 0.02, SCREEN_HEIGHT * 0.02)
    score_text_surf = font.render(f"Score: {int(score)}", True, WHITE)
    SCREEN.blit(score_text_surf, (10, 10))

    if not game_active:
        game_over_message = "Game Over!"
        restart_message = "Tap or Space to Restart" # Updated for touch/mouse
        
        game_over_text_surf = font.render(game_over_message, True, WHITE)
        restart_text_surf = font.render(restart_message, True, WHITE)
        
        game_over_rect = game_over_text_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        restart_rect = restart_text_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        
        SCREEN.blit(game_over_text_surf, game_over_rect)
        SCREEN.blit(restart_text_surf, restart_rect)

    pygame.display.flip()
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
sys.exit()
