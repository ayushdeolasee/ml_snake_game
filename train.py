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
FREQ_OF_TARGET_NN_UPDATE = 750 
REPLAY_MEMORY_SIZE = 15500000
HIDDEN_SIZE = 5128 
EPOCHS = 1000000
BATCH_SIZE = 1000 

r = redis.Redis(host="localhost", port=6379, db=0)
ping = r.ping()

if ping:
    print("Succesfully connected to redis instance")
else:
    print("Error occured when connecting to redis instance")


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device}")

def policy(epsilon, action_space, highest_estimated_action):
    if random.random() < epsilon:
        return random.randint(0, len(action_space) - 1)
    else: 
        return highest_estimated_action

def loss_function(active_prediction, gamma, immediate_reward, action, done, final_state, target_nn):
    active_prediction = active_prediction.gather(dim=1, index=action).squeeze(1)
    next_state_q_values = target_nn(final_state) 
    next_state_value_target = torch.max(next_state_q_values, dim=1)[0]
    q_target = immediate_reward + (1.0 - done) * (gamma * next_state_value_target) 

    loss = (q_target - active_prediction) ** 2
    return loss.mean()

#[head_x, head_y, direction, food_x, food_y, danger_straight, danger_right, danger_left]
class DQN_scaled_ff(nn.Module):
    def __init__(self, input_size, output_size, hidden_size) -> None:
        super().__init__()
        hidden_size1 = int(hidden_size * 2)
        hidden_size2 = int(hidden_size / 2)
        
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.linear_sequence1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size1),
            nn.ReLU(), 
            nn.Linear(hidden_size1, hidden_size1),
            nn.Linear(hidden_size1, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size1),
            nn.Linear(hidden_size1, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size1)
        )
        self.linear_sequence2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size2),
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(), 
            nn.Linear(hidden_size2, hidden_size2),
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(), 
            nn.Linear(hidden_size2, hidden_size2)
        )
        self.output_linear = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.input_linear(x) 
        x = self.linear_sequence1(x)
        x = self.linear_sequence2(x)
        x = self.output_linear(x)
        return x
    
    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

def train(redis, active_nn, target_nn, device, loss_function, optimizer, gamma, epochs, freq_of_target_nn_update, epsilon, action_space, policy, model_name, model_type, learning_rate, run_name, batch_size, save_model=False):
    replay_memory_size = redis.dbsize()
    wandb.init(
        project="snake_game_rl",
        name=run_name,
        config={
            "model_name": model_name,
            "model_type": model_type,
            "epsilon": epsilon,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "replay_memory_size": replay_memory_size,
            "epochs": epochs,
            "freq_of_target_nn_update": freq_of_target_nn_update,
            "batch_size": batch_size,
        }
    )
    
    training_table = wandb.Table(columns=["epoch", "loss"])
    evaluation_table = wandb.Table(columns=["epoch", "steps", "score"])


    for epoch in range(epochs):
        initial_state = []
        final_state = []
        action = []
        action_pred = []
        reward = []
        done = []

        replay_memory_size = r.dbsize()
        random_keys = random.sample(range(0, replay_memory_size - 1), k=batch_size)

        for key in random_keys:
            experience = pickle.loads(r.get(key))
            initial_state += [experience[0]]
            action += [experience[1]]
            final_state += [experience[2]]
            reward += [experience[3]]
            done += [experience[4]]

        initial_state = torch.tensor(initial_state, dtype=torch.float32, device="mps")
        final_state = torch.tensor(final_state, dtype=torch.float32, device="mps")
        action = torch.tensor(action, device="mps", dtype=torch.long).unsqueeze(0).T
        reward = torch.tensor(reward, dtype=torch.float32, device="mps")
        done = torch.tensor(done, dtype=torch.float32, device="mps")

        optimizer.zero_grad() 
        active_nn.train() 
        action_pred = active_nn(initial_state)
        loss = loss_function(action_pred, gamma, reward, action, done, final_state, target_nn)
    
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        
        wandb.log({
            "epoch": epoch,
            "training_loss": loss_value
        })
        
        training_table.add_data(str(epoch), str(loss_value))
        
        print(f"Epoch {epoch}, Loss: {loss_value}")
    
        if epoch % freq_of_target_nn_update == 0:
            target_nn.load_state_dict(active_nn.state_dict())
            target_nn.requires_grad_ = False

            print("Evaluating active network")
            steps = 0
            score = 0
            evaluation_game = Snake(evaluation=True)
            
            with torch.no_grad():
                active_nn.eval() 
                while True:
                    initial_state = evaluation_game.get_state()
                    action_pred = active_nn(torch.tensor(initial_state, dtype=torch.float32).to(device))
                    action = policy(0, action_space, torch.argmax(action_pred).item())
                    done = evaluation_game.move_with_action(action=action)
                    steps += 1
                
                    if evaluation_game.get_immediate_reward() == 1:
                        score += 1
                
                    if done == 1:
                        break
                    if steps > 5000:
                        break

            wandb.log({
                "epoch": epoch,
                "evaluation_steps": steps,
                "evaluation_score": score
            })
            
            evaluation_table.add_data(str(epoch), str(steps), str(score))
            
            print(f"Evaluation - Steps: {steps}, Score: {score}")
    
    wandb.log({"training_metrics_table": training_table})
    wandb.log({"evaluation_metrics_table": evaluation_table})
    
    summary_data = [
        ["Model Name", model_name],
        ["Model Type", model_type],
        ["Epsilon", str(epsilon)],
        ["Gamma", str(gamma)],
        ["Learning Rate", str(learning_rate)],
        ["Replay Memory Size", str(replay_memory_size)],
        ["Epochs", str(epochs)],
        ["Target NN Update Frequency", str(freq_of_target_nn_update)],
        ["Final Loss", str(loss_value)],
        ["Final Evaluation Steps", str(steps) if 'steps' in locals() else 'N/A'],
        ["Final Evaluation Score", str(score) if 'score' in locals() else 'N/A']
    ]
    
    summary_table = wandb.Table(columns=["Parameter", "Value"], data=summary_data)
    wandb.log({"training_summary_table": summary_table})
    
    if save_model:
        active_nn.save(f"{model_name}.pth")
    
    wandb.finish()
    
    return active_nn, target_nn


active_nn = DQN_scaled_ff(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE).to(device=device)
target_nn = DQN_scaled_ff(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE).to(device=device)
target_nn.load_state_dict(active_nn.state_dict())
target_nn.requires_grad_ = False
optimizer = optim.AdamW(active_nn.parameters(), LR)

active_nn, target_nn = train(
    redis=r, 
    active_nn=active_nn,
    target_nn=target_nn,
    device=device,
    loss_function=loss_function,
    optimizer=optimizer,
    gamma=GAMMA,
    epochs=EPOCHS,
    freq_of_target_nn_update=FREQ_OF_TARGET_NN_UPDATE,
    epsilon=EPSILON,
    action_space=ACTION_SPACE,
    batch_size=BATCH_SIZE,
    policy=policy,
    model_name="scaled_ff_model_batched",
    model_type="scaled_ff",
    learning_rate=LR,
    run_name="scaled_ff_model_run_batched_larger",
    save_model=True
)

from game import Snake
steps = 0
score = 0

evaluation_game = Snake(evaluation=True)
while True:
    initial_state = evaluation_game.get_state()
    action_pred = active_nn(torch.tensor(initial_state, dtype=torch.float32).to(device))
    action = policy(0, ACTION_SPACE, torch.argmax(action_pred).item())
    done = evaluation_game.move_with_action(action=action)
    steps += 1
    if evaluation_game.get_immediate_reward() == 1:
        score += 1
    
    if done == 1:
        break


print(f"Evaluation steps: {steps}, score: {score}")
