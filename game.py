import pygame
import sys
import random
import time
from pygame import gfxdraw

# Initialize pygame
pygame.init()
pygame.font.init()

# Game constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 205, 50)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 100, 0)
SNAKE_COLOR = (46, 139, 87)
FOOD_COLOR = (220, 20, 60)
BG_COLOR = (240, 255, 240)
GRID_COLOR = (220, 240, 220)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 25)
large_font = pygame.font.SysFont('Arial', 50)

class Snake:
    def __init__(self):
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.grow = False
        self.color = SNAKE_COLOR
        
    def get_head_position(self):
        return self.positions[0]
    
    def turn(self, direction):
        # Use absolute directions instead of relative turning
        if direction == "up":
            # Only allow up if not currently moving down (prevent immediate self-collision)
            if self.direction != DOWN:
                self.direction = UP
        elif direction == "down":
            # Only allow down if not currently moving up
            if self.direction != UP:
                self.direction = DOWN
        elif direction == "left":
            # Only allow left if not currently moving right
            if self.direction != RIGHT:
                self.direction = LEFT
        elif direction == "right":
            # Only allow right if not currently moving left
            if self.direction != LEFT:
                self.direction = RIGHT
    
    def move(self):
        head = self.get_head_position()
        x, y = self.direction
        new_head = (head[0] + x, head[1] + y)
        
        # Check for wall collision
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            return True  # Collision detected
            
        # Check for self collision
        if new_head in self.positions[1:]:
            return True  # Collision detected
        
        self.positions.insert(0, new_head)
        if not self.grow:
            self.positions.pop()
        else:
            self.grow = False
        return False  # No collision
    
    def grow_snake(self):
        self.grow = True
    
    def draw(self, surface):
        for i, pos in enumerate(self.positions):
            # Create gradient color effect from head to tail
            r = max(20, self.color[0] - i * 0.5)
            g = max(100, self.color[1] - i * 0.5)
            b = max(20, self.color[2] - i * 0.5)
            segment_color = (r, g, b)
            
            rect = pygame.Rect(pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(surface, segment_color, rect)
            pygame.draw.rect(surface, DARK_GREEN, rect, 1)  # Border
            
            # Draw eyes on the head
            if i == 0:
                # Determine eye positions based on direction
                if self.direction == UP:
                    left_eye = (rect.left + rect.width * 0.25, rect.top + rect.height * 0.25)
                    right_eye = (rect.left + rect.width * 0.75, rect.top + rect.height * 0.25)
                elif self.direction == DOWN:
                    left_eye = (rect.left + rect.width * 0.25, rect.top + rect.height * 0.75)
                    right_eye = (rect.left + rect.width * 0.75, rect.top + rect.height * 0.75)
                elif self.direction == LEFT:
                    left_eye = (rect.left + rect.width * 0.25, rect.top + rect.height * 0.25)
                    right_eye = (rect.left + rect.width * 0.25, rect.top + rect.height * 0.75)
                else:  # RIGHT
                    left_eye = (rect.left + rect.width * 0.75, rect.top + rect.height * 0.25)
                    right_eye = (rect.left + rect.width * 0.75, rect.top + rect.height * 0.75)
                
                pygame.draw.circle(surface, WHITE, (int(left_eye[0]), int(left_eye[1])), 3)
                pygame.draw.circle(surface, WHITE, (int(right_eye[0]), int(right_eye[1])), 3)
                pygame.draw.circle(surface, BLACK, (int(left_eye[0]), int(left_eye[1])), 1)
                pygame.draw.circle(surface, BLACK, (int(right_eye[0]), int(right_eye[1])), 1)

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = FOOD_COLOR
        self.randomize_position([])  # Initialize with empty snake positions
    
    def randomize_position(self, snake_positions):
        # Create a list of all possible grid positions
        all_positions = [(x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)]
        
        # Remove positions occupied by the snake
        available_positions = [pos for pos in all_positions if pos not in snake_positions]
        
        # If there are no available positions (snake fills the grid), just return
        if not available_positions:
            return
            
        # Choose a random position from available ones
        self.position = random.choice(available_positions)
    
    def draw(self, surface):
        rect = pygame.Rect(self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, 
                          GRID_SIZE, GRID_SIZE)
        
        # Draw a fancy apple-like food
        pygame.draw.circle(surface, self.color, rect.center, GRID_SIZE // 2)
        
        # Stem
        stem_rect = pygame.Rect(rect.centerx - 1, rect.top, 2, GRID_SIZE // 4)
        pygame.draw.rect(surface, DARK_GREEN, stem_rect)
        
        # Shine effect
        shine_pos = (rect.centerx - GRID_SIZE // 4, rect.centery - GRID_SIZE // 4)
        pygame.draw.circle(surface, (255, 220, 220), shine_pos, 2)

def draw_grid():
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

def show_score(score):
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))

def game_over(score):
    screen.fill(BG_COLOR)
    
    game_over_text = large_font.render("Game Over", True, RED)
    score_text = font.render(f"Final Score: {score}", True, BLACK)
    restart_text = font.render("Press R to Restart or Q to Quit", True, BLACK)
    
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 3))
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2))
    screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 1.5))
    
    pygame.display.update()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True  # Restart
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

def main():
    global FPS
    snake = Snake()
    food = Food()
    food.randomize_position(snake.positions)  # Initialize food position correctly
    score = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.turn("up")
                elif event.key == pygame.K_DOWN:
                    snake.turn("down")
                elif event.key == pygame.K_LEFT:
                    snake.turn("left")
                elif event.key == pygame.K_RIGHT:
                    snake.turn("right")
        
        # Move snake and check for collisions
        collision = snake.move()
        if collision:
            if game_over(score):
                # Restart game
                snake = Snake()
                food = Food()
                food.randomize_position(snake.positions)  # Reset food position
                score = 0
                FPS = 10  # Reset FPS when restarting
        
        # Check if snake ate food
        if snake.get_head_position() == food.position:
            snake.grow_snake()
            food.randomize_position(snake.positions)  # Update this line
            score += 10
            # Increase speed slightly with each food eaten
            FPS = min(20, 10 + score // 50)  # Cap at 20 FPS
        
        # Draw everything
        screen.fill(BG_COLOR)
        draw_grid()
        snake.draw(screen)
        food.draw(screen)
        show_score(score)
        
        pygame.display.update()
        clock.tick(FPS)

if __name__ == "__main__":
    main()