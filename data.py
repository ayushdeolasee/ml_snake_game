import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from game import Snake
import os
import shutil
import wandb
import redis
import pickle
import shutil
import random
from collections import deque

EPSILON = 1 
EPSILON_DECAY = 0.095
ACTION_SPACE = [0,1,2,3,4]
GAMMA = 0.99
LR = 1e-3
INPUT_SIZE = 8
OUTPUT_SIZE = 5 
FREQ_OF_TARGET_NN_UPDATE = 20
REPLAY_MEMORY_SIZE = 15500000
HIDDEN_SIZE = 64 
EPOCHS = 1000000
EPISODE_ID = 0

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device}")

r = redis.Redis(host="localhost", port=6379, db=0)
ping = r.ping()

if ping:
    print("Succesfully connected to redis instance")
else:
    print("Error occured when connecting to redis instance")

def policy(epsilon, action_space, highest_estimated_action):
    if random.random() < epsilon:
        return random.randint(0, len(action_space) - 1)
    else: 
        return highest_estimated_action

#[head_x, head_y, direction, food_x, food_y, danger_straight, danger_right, danger_left]
class DQN_base_ff(nn.Module):
    def __init__(self, input_size, output_size, hidden_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# active_nn = DQN_scaled_ff(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE).to(device=device)
active_nn = DQN_base_ff(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE).to(device=device)
snapshot_name = "50000000_size.rdb"

game = Snake()
for episode in range(REPLAY_MEMORY_SIZE):
    initial_state = game.get_state()
    action_pred = torch.argmax(active_nn(torch.tensor(initial_state, dtype=torch.float32).to(device))).item()
    action = policy(EPSILON, ACTION_SPACE, action_pred)
    done = game.move_with_action(action=action)
    final_state = game.get_state()
    reward = game.get_immediate_reward()
    
    experience = (initial_state, action, final_state, reward, done)
    r.set(EPISODE_ID, pickle.dumps(experience))
    EPISODE_ID += 1

    if done == 1:
        game = Snake()

import os

data_dir = "data"
snapshot_path = os.path.join(data_dir, snapshot_name)

if not os.path.exists(snapshot_path):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    r.save()  
    redis_config = r.config_get()
    dbfilename = redis_config.get("dbfilename", "dump.rdb")
    dirpath = redis_config.get("dir", ".")
    default_dump_path = os.path.join(dirpath, dbfilename)
    shutil.copy(default_dump_path, snapshot_path)
