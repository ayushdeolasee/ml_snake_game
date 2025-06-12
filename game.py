import random

GRID_WIDTH = 40  # 800 // 20
GRID_HEIGHT = 30  # 600 // 20

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self, evaluation=False):
        if evaluation:
            # Start at the center for evaluation
            start_x = GRID_WIDTH // 2
            start_y = GRID_HEIGHT // 2
        else:
            # Randomize starting position for training
            start_x = random.randint(0, GRID_WIDTH - 1)
            start_y = random.randint(0, GRID_HEIGHT - 1)
        self.positions = [(start_x, start_y)]
        self.direction = RIGHT
        self.grow = False
        self.reward = 0 
        self.immediate_reward = 0
        self.food = Food()  # Automatically place food
        self.food.randomize_position(self.positions)

    def get_head_position(self):
        return self.positions[0]

    def get_food_position(self):
        return self.food.position

    def get_reward(self):
        return self.reward

    def get_immediate_reward(self):
        return self.immediate_reward

    def move_with_action(self, action):
        # Reset immediate reward at start of each move
        self.immediate_reward = 0
        
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
        return self.move()

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
    
    def _get_right_direction(self):
        """Get the direction that is to the right of current direction"""
        if self.direction == UP:
            return RIGHT
        elif self.direction == RIGHT:
            return DOWN
        elif self.direction == DOWN:
            return LEFT
        else:  # LEFT
            return UP
    
    def _get_left_direction(self):
        """Get the direction that is to the left of current direction"""
        if self.direction == UP:
            return LEFT
        elif self.direction == LEFT:
            return DOWN
        elif self.direction == DOWN:
            return RIGHT
        else:  # RIGHT
            return UP

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