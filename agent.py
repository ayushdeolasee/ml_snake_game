import random
from game import Snake
import torch
import torch.nn as nn

EPSILON = 1 
ACTION_SPACE = [0,1,2,3]

def policy(epsilon, action_space, highest_estimated_action):
    if random.random() < epsilon:
        return random.randint(0, len(action_space) - 1)
    else: 
        return highest_estimated_action

#[head_x, head_y, direction, food_x, food_y, danger_straight, danger_right, danger_left]

class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass
    def foward():
        pass


def train():
    pass