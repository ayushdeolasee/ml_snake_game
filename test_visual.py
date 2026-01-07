#!/usr/bin/env python3
"""
Test script to demonstrate the visual Snake game functionality.
Run this to see the snake playing with a visual grid display.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from game import Snake
import random

# Simple DQN for demonstration
class SimpleDQN(nn.Module):
    def __init__(self, input_size=8, output_size=5, hidden_size=32):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def random_policy(action_space):
    """Random policy for demonstration"""
    return random.randint(0, len(action_space) - 1)

def simple_policy(state):
    """Simple rule-based policy for demonstration"""
    head_x, head_y, direction, food_x, food_y, danger_straight, danger_right, danger_left = state
    
    # If there's danger straight ahead, turn
    if danger_straight:
        if not danger_right:
            return 4 if direction == 0 else 1 if direction == 3 else 3 if direction == 1 else 2  # Turn right relative to current direction
        elif not danger_left:
            return 3 if direction == 0 else 2 if direction == 3 else 4 if direction == 1 else 1   # Turn left relative to current direction
        else:
            return random.randint(1, 4)  # Random turn if trapped
    
    # Try to move towards food
    if food_x > head_x and direction != 2:  # Food is to the right
        return 4  # Turn right
    elif food_x < head_x and direction != 3:  # Food is to the left  
        return 3  # Turn left
    elif food_y < head_y and direction != 1:  # Food is above
        return 1  # Turn up
    elif food_y > head_y and direction != 0:  # Food is below
        return 2  # Turn down
    
    return 0  # Continue straight

def test_visual_mode():
    """Test the visual mode of the snake game"""
    print("🐍 Testing Snake Game Visual Mode")
    print("=" * 50)
    print("This will show the AI snake playing with visual display.")
    print("Press Ctrl+C to stop at any time.")
    print("Starting in 3 seconds...")
    
    import time
    time.sleep(3)
    
    # Create snake in evaluation mode (this will automatically enable visual display)
    game = Snake(evaluation=True)
    
    steps = 0
    score = 0
    
    try:
        while True:
            # Get current state
            state = game.get_state()
            
            # Use simple policy to choose action
            action = simple_policy(state)
            
            # Move snake with visual display (step_count and delay for animation)
            done = game.move_with_action(action, step_count=steps, delay=0.2)
            
            steps += 1
            
            # Update score if food was eaten
            if game.get_immediate_reward() == 1:
                score += 1
            
            # Check if game ended
            if done == 1:
                game.display_grid(steps, score, "GAME OVER!")
                print(f"\n🎮 Game Over!")
                print(f"📊 Final Stats:")
                print(f"   Steps: {steps}")
                print(f"   Score: {score}")
                print(f"   Snake Length: {len(game.positions)}")
                break
                
            # Optional: limit game length for demonstration
            if steps > 200:
                game.display_grid(steps, score, "Demo Complete")
                print(f"\n✅ Demo completed after {steps} steps!")
                print(f"📊 Final Stats:")
                print(f"   Steps: {steps}")
                print(f"   Score: {score}")
                print(f"   Snake Length: {len(game.positions)}")
                break
                
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Game stopped by user")
        print(f"📊 Final Stats:")
        print(f"   Steps: {steps}")
        print(f"   Score: {score}")
        print(f"   Snake Length: {len(game.positions)}")

def test_with_neural_network():
    """Test with a simple neural network (random weights)"""
    print("\n🧠 Testing with Neural Network (Random Weights)")
    print("=" * 50)
    
    # Create a simple neural network with random weights
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = SimpleDQN().to(device)
    
    game = Snake(evaluation=True)
    steps = 0
    score = 0
    
    try:
        while steps < 100:  # Limit to 100 steps for demo
            state = game.get_state()
            
            # Get action from neural network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                action_values = net(state_tensor)
                action = torch.argmax(action_values).item()
            
            done = game.move_with_action(action, step_count=steps, delay=0.15)
            steps += 1
            
            if game.get_immediate_reward() == 1:
                score += 1
                
            if done == 1:
                game.display_grid(steps, score, "NEURAL NETWORK - GAME OVER!")
                break
                
    except KeyboardInterrupt:
        print(f"\n⏹️  Neural network demo stopped")
        
    print(f"\n🧠 Neural Network Demo Stats:")
    print(f"   Steps: {steps}")
    print(f"   Score: {score}")
    print(f"   Snake Length: {len(game.positions)}")

if __name__ == "__main__":
    print("🐍 Snake Game Visual Demo")
    print("=" * 50)
    print("Choose demo mode:")
    print("1. Simple rule-based AI")
    print("2. Random neural network")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            test_visual_mode()
        elif choice == "2":
            test_with_neural_network()
        elif choice == "3":
            test_visual_mode()
            input("\nPress Enter to continue to neural network demo...")
            test_with_neural_network()
        else:
            print("Invalid choice, running simple demo...")
            test_visual_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}") 
