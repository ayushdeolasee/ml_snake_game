import random
import os
import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Visual mode will use terminal display.")

GRID_WIDTH = 40  # 800 // 20
GRID_HEIGHT = 30  # 600 // 20
CELL_SIZE = 20  # Size of each grid cell in pixels
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + 100  # Extra space for UI

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 128, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self, evaluation=False, show_visual=True):
        self.evaluation = evaluation
        self.show_visual = show_visual
        
        if evaluation:
            # Start at the center for evaluation, length 1
            start_x = GRID_WIDTH // 2
            start_y = GRID_HEIGHT // 2
            self.positions = [(start_x, start_y)]
            self.direction = RIGHT
        else:
            # Randomize starting position and length for training
            self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
            if self.direction == RIGHT:
                max_length = min(100, GRID_WIDTH)
                start_x = random.randint(0, GRID_WIDTH - 1)
                start_y = random.randint(0, GRID_HEIGHT - 1)
                max_possible = start_x + 1
                initial_length = random.randint(1, min(max_length, max_possible))
                self.positions = [(start_x - i, start_y) for i in range(initial_length)][::-1]
            elif self.direction == LEFT:
                max_length = min(100, GRID_WIDTH)
                start_x = random.randint(0, GRID_WIDTH - 1)
                start_y = random.randint(0, GRID_HEIGHT - 1)
                max_possible = GRID_WIDTH - start_x
                initial_length = random.randint(1, min(max_length, max_possible))
                self.positions = [(start_x + i, start_y) for i in range(initial_length)][::-1]
            elif self.direction == DOWN:
                max_length = min(100, GRID_HEIGHT)
                start_x = random.randint(0, GRID_WIDTH - 1)
                start_y = random.randint(0, GRID_HEIGHT - 1)
                max_possible = start_y + 1
                initial_length = random.randint(1, min(max_length, max_possible))
                self.positions = [(start_x, start_y - i) for i in range(initial_length)][::-1]
            else:  # UP
                max_length = min(100, GRID_HEIGHT)
                start_x = random.randint(0, GRID_WIDTH - 1)
                start_y = random.randint(0, GRID_HEIGHT - 1)
                max_possible = GRID_HEIGHT - start_y
                initial_length = random.randint(1, min(max_length, max_possible))
                self.positions = [(start_x, start_y + i) for i in range(initial_length)][::-1]
        
        self.grow = False
        self.reward = 0 
        self.immediate_reward = 0
        self.food = Food()  # Automatically place food
        self.food.randomize_position(self.positions)
        
        # Initialize pygame display if evaluation mode is enabled and visuals are on
        if self.evaluation and self.show_visual and PYGAME_AVAILABLE:
            self.init_pygame()
        elif self.evaluation and self.show_visual:
            self.clear_screen()
            
            print("🐍 Snake Game - AI Evaluation Mode (Terminal)")
            print("=" * 50)

    def init_pygame(self):
        """Initialize pygame display"""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI - Evaluation Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_pygame(self, step_count=0, score=0, action_name="", fps=10):
        """Display the game using pygame"""
        if not self.evaluation or not self.show_visual or not PYGAME_AVAILABLE:
            return
            
        # Handle pygame events to prevent window freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        # Clear screen
        self.screen.fill(BLACK)
        
        # Draw grid
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, GRID_HEIGHT * CELL_SIZE))
        for y in range(0, GRID_HEIGHT * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (WINDOW_WIDTH, y))
        
        # Draw food
        food_x, food_y = self.food.position
        food_rect = pygame.Rect(food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, RED, food_rect)
        pygame.draw.rect(self.screen, WHITE, food_rect, 2)
        
        # Draw snake
        for i, (x, y) in enumerate(self.positions):
            snake_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if i == 0:  # Head
                pygame.draw.rect(self.screen, DARK_GREEN, snake_rect)
                pygame.draw.rect(self.screen, WHITE, snake_rect, 3)
                
                # Draw direction arrow on head
                center_x = x * CELL_SIZE + CELL_SIZE // 2
                center_y = y * CELL_SIZE + CELL_SIZE // 2
                arrow_size = CELL_SIZE // 3
                
                if self.direction == UP:
                    pygame.draw.polygon(self.screen, WHITE, [
                        (center_x, center_y - arrow_size),
                        (center_x - arrow_size//2, center_y + arrow_size//2),
                        (center_x + arrow_size//2, center_y + arrow_size//2)
                    ])
                elif self.direction == DOWN:
                    pygame.draw.polygon(self.screen, WHITE, [
                        (center_x, center_y + arrow_size),
                        (center_x - arrow_size//2, center_y - arrow_size//2),
                        (center_x + arrow_size//2, center_y - arrow_size//2)
                    ])
                elif self.direction == LEFT:
                    pygame.draw.polygon(self.screen, WHITE, [
                        (center_x - arrow_size, center_y),
                        (center_x + arrow_size//2, center_y - arrow_size//2),
                        (center_x + arrow_size//2, center_y + arrow_size//2)
                    ])
                elif self.direction == RIGHT:
                    pygame.draw.polygon(self.screen, WHITE, [
                        (center_x + arrow_size, center_y),
                        (center_x - arrow_size//2, center_y - arrow_size//2),
                        (center_x - arrow_size//2, center_y + arrow_size//2)
                    ])
            else:  # Body
                pygame.draw.rect(self.screen, GREEN, snake_rect)
                pygame.draw.rect(self.screen, WHITE, snake_rect, 1)
        
        # Draw UI information
        ui_y = GRID_HEIGHT * CELL_SIZE + 10
        
        # Game stats
        step_text = self.font.render(f"Step: {step_count}", True, WHITE)
        score_text = self.font.render(f"Score: {score}", True, WHITE)
        length_text = self.font.render(f"Length: {len(self.positions)}", True, WHITE)
        
        self.screen.blit(step_text, (10, ui_y))
        self.screen.blit(score_text, (150, ui_y))
        self.screen.blit(length_text, (250, ui_y))
        
        # Action information
        if action_name:
            action_text = self.small_font.render(f"Action: {action_name}", True, LIGHT_GRAY)
            self.screen.blit(action_text, (10, ui_y + 40))
        
        # Instructions
        instruction_text = self.small_font.render("Press ESC or close window to stop", True, LIGHT_GRAY)
        self.screen.blit(instruction_text, (10, ui_y + 65))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(fps)
        
        return True

    def display_grid(self, step_count=0, score=0, action_name=""):
        """Display the current game state as a grid (terminal fallback)"""
        if not self.evaluation or not self.show_visual:
            return
            
        # Clear screen for animation effect
        self.clear_screen()
        
        # Print header information
        print("🐍 Snake Game - AI Playing (Terminal Mode)")
        print("=" * 50)
        print(f"Step: {step_count} | Score: {score} | Length: {len(self.positions)}")
        if action_name:
            print(f"Action: {action_name}")
        print("=" * 50)
        
        # Create the grid
        grid = [[' ' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # Place food
        food_x, food_y = self.food.position
        grid[food_y][food_x] = '🍎'
        
        # Place snake
        for i, (x, y) in enumerate(self.positions):
            if i == 0:  # Head
                # Show direction with arrow
                if self.direction == UP:
                    grid[y][x] = '↑'
                elif self.direction == DOWN:
                    grid[y][x] = '↓'
                elif self.direction == LEFT:
                    grid[y][x] = '←'
                else:  # RIGHT
                    grid[y][x] = '→'
            else:  # Body
                grid[y][x] = '●'
        
        # Print the grid with borders
        print('┌' + '─' * GRID_WIDTH + '┐')
        for row in grid:
            print('│' + ''.join(row) + '│')
        print('└' + '─' * GRID_WIDTH + '┘')
        
        # Print controls info
        print("\nControls: 0=straight, 1=up, 2=down, 3=left, 4=right")
        print("Press Ctrl+C to stop")

    def get_action_name(self, action):
        """Convert action number to readable name"""
        action_names = {
            0: "Continue Straight",
            1: "Turn Up", 
            2: "Turn Down",
            3: "Turn Left",
            4: "Turn Right"
        }
        return action_names.get(action, "Unknown")

    def get_head_position(self):
        return self.positions[0]

    def get_food_position(self):
        return self.food.position

    def get_reward(self):
        return self.reward

    def get_immediate_reward(self):
        return self.immediate_reward

    def move_with_action(self, action, step_count=0, step_penalty=0.01, delay=0.1, fps=10, alpha=0.01):
        # Reset immediate reward at start of each move
        self.immediate_reward = 0
        food_x, food_y = self.get_food_position()
        head_x, head_y = self.get_head_position()
        self.current_distance = abs(food_x - head_x) + abs(food_y - head_y)

        # Display current state before move (if in evaluation mode and visuals are enabled)
        if self.evaluation and self.show_visual:
            action_name = self.get_action_name(action)
            
            if PYGAME_AVAILABLE:
                # Use pygame display
                if not self.display_pygame(step_count, self.reward, action_name, fps):
                    return 1  # User closed window
            else:
                # Fallback to terminal display
                self.display_grid(step_count, self.reward, action_name)
                time.sleep(delay)
        
        # action: 0=straight, 1=up, 2=down, 3=left, 4=right
        if action == 1 and self.direction != DOWN:
            self.direction = UP
        elif action == 2 and self.direction != UP:
            self.direction = DOWN
        elif action == 3 and self.direction != RIGHT:
            self.direction = LEFT
        elif action == 4 and self.direction != LEFT:
            self.direction = RIGHT
        # else action == 0: keep current direction
        reward = self.move()
        # Recompute distance using UPDATED head position
        
        new_head_x, new_head_y = self.get_head_position()
        self.new_distance = abs(food_x - new_head_x) + abs(food_y - new_head_y)
        self.distance_reward = (self.current_distance - self.new_distance) * (self.new_distance)
        # self.new_distance = 1.0 / (self.new_distance + 1.0) 
        # self.distance_reward = (self.new_distance - self.current_distance)
        # self.distance_reward = 1/(self.new_distance + 1)
        # # self.distance_reward -= (step_penalty * step_count)
        self.immediate_reward += alpha * self.distance_reward
        return reward

    def move(self):
        head = self.get_head_position()
        x, y = self.direction
        new_head = (head[0] + x, head[1] + y)
        
        # Check for wall collision
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.reward -= 1
            self.immediate_reward = -1 
            return 1
            
        # Check for self collision
        if new_head in self.positions[1:]:
            self.reward -= 1
            self.immediate_reward = -1 
            return 1  
        
        self.positions.insert(0, new_head)
        if not self.grow:
            self.positions.pop()
        else:
            self.grow = False

        # Check if snake eats food
        if self.get_head_position() == self.food.position:
            self.grow_snake()
            self.food.randomize_position(self.positions)
        return 0  # No collision
    
    def grow_snake(self):
        self.reward += 1
        self.immediate_reward = 1 
        self.grow = True

    def get_state(self):
        head_x, head_y = self.get_head_position()
        food_x, food_y = self.food.position
        
        if self.direction == UP:
            dir_idx = 0
        elif self.direction == DOWN:
            dir_idx = 1
        elif self.direction == LEFT:
            dir_idx = 2
        else:  # RIGHT
            dir_idx = 3
            
        # Danger detection - check for collision in each direction
        danger_straight = self._is_collision(head_x + self.direction[0], head_y + self.direction[1])
        danger_right = self._is_collision_direction(self._get_right_direction())
        danger_left = self._is_collision_direction(self._get_left_direction())
        
        return [head_x, head_y, dir_idx, food_x, food_y, int(danger_straight), int(danger_right), int(danger_left)]
    
    def _is_collision(self, x, y):
        """Check if position (x,y) would result in collision"""
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return True
        if (x, y) in self.positions:
            return True
        return False
    
    def _is_collision_direction(self, direction):
        """Check if moving in given direction would cause collision"""
        head_x, head_y = self.get_head_position()
        new_x = head_x + direction[0]
        new_y = head_y + direction[1]
        return self._is_collision(new_x, new_y)
    
    def _rotate_direction(self, base_direction, turn):
        """Rotate a direction 90 degrees to 'left' or 'right' relative to base_direction"""
        dx, dy = base_direction
        if turn == 'right':
            return (dy, -dx)
        elif turn == 'left':
            return (-dy, dx)
        return base_direction
    
    def _get_right_direction(self):
        """Get the direction that is to the right of current direction"""
        return self._rotate_direction(self.direction, 'right')
    
    def _get_left_direction(self):
        """Get the direction that is to the left of current direction"""
        return self._rotate_direction(self.direction, 'left')

    def cleanup(self):
        """Clean up pygame resources"""
        if self.evaluation and self.show_visual and PYGAME_AVAILABLE:
            pygame.quit()

class Food:
    def __init__(self):
        self.position = (0, 0)
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
