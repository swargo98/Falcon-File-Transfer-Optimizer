import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
from gym import spaces
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_model(agent, filename_policy, filename_value):
    torch.save(agent.policy.state_dict(), filename_policy)
    torch.save(agent.value_function.state_dict(), filename_value)
    print("Model saved successfully.")


def load_model(agent, filename_policy, filename_value):
    agent.policy.load_state_dict(torch.load(filename_policy))
    agent.policy_old.load_state_dict(agent.policy.state_dict())
    agent.value_function.load_state_dict(torch.load(filename_value))
    print("Model loaded successfully.")

class SimulatorState:
    def __init__(self, sender_buffer_remaining_capacity=0, receiver_buffer_remaining_capacity=0,
                 read_throughput=0, write_throughput=0, network_throughput=0,
                 read_thread=0, write_thread=0, network_thread=0) -> None:
        self.sender_buffer_remaining_capacity = sender_buffer_remaining_capacity
        self.receiver_buffer_remaining_capacity = receiver_buffer_remaining_capacity
        self.read_throughput = read_throughput
        self.write_throughput = write_throughput
        self.network_throughput = network_throughput
        self.read_thread = read_thread
        self.write_thread = write_thread
        self.network_thread = network_thread

    def copy(self):
        # Return a new SimulatorState instance with the same attribute values
        return SimulatorState(
            sender_buffer_remaining_capacity=self.sender_buffer_remaining_capacity,
            receiver_buffer_remaining_capacity=self.receiver_buffer_remaining_capacity,
            read_throughput=self.read_throughput,
            write_throughput=self.write_throughput,
            network_throughput=self.network_throughput,
            read_thread=self.read_thread,
            write_thread=self.write_thread,
            network_thread=self.network_thread
        )

    def to_array(self):
        # Convert the state to a NumPy array
        return np.array([
            self.sender_buffer_remaining_capacity,
            self.receiver_buffer_remaining_capacity,
            self.read_throughput,
            self.write_throughput,
            self.network_throughput,
            self.read_thread,
            self.write_thread,
            self.network_thread
        ], dtype=np.float32)

from typing_extensions import final
class NetworkSystemSimulator:
    def __init__(self, read_thread = 1, network_thread = 1, write_thread = 1, sender_buffer_capacity = 10, receiver_buffer_capacity = 10, read_throughput_per_thread = 3, write_throughput_per_thread = 1, network_throughput_per_thread = 2, read_bandwidth = 6, write_bandwidth = 6, network_bandwidth = 6, read_background_traffic = 0, write_background_traffic = 0, network_background_traffic = 0, track_states = False):
        self.sender_buffer_capacity = sender_buffer_capacity
        self.receiver_buffer_capacity = receiver_buffer_capacity
        self.read_throughput_per_thread = read_throughput_per_thread
        self.write_throughput_per_thread = write_throughput_per_thread
        self.network_throughput_per_thread = network_throughput_per_thread
        self.read_bandwidth = read_bandwidth
        self.write_bandwidth = write_bandwidth
        self.network_bandwidth = network_bandwidth
        self.read_background_traffic = read_background_traffic
        self.write_background_traffic = write_background_traffic
        self.network_background_traffic = network_background_traffic
        self.read_thread = read_thread
        self.network_thread = network_thread
        self.write_thread = write_thread
        self.track_states = track_states
        self.K = 1.02

        # Initialize the buffers
        self.sender_buffer_in_use = max(min(self.read_throughput_per_thread * read_thread - self.network_throughput_per_thread * self.network_thread, self.sender_buffer_capacity), 0)
        self.receiver_buffer_in_use = max(min(self.network_throughput_per_thread * network_thread - self.write_throughput_per_thread * self.write_thread, self.receiver_buffer_capacity), 0)

        print(f"Initial Sender Buffer: {self.sender_buffer_in_use}, Receiver Buffer: {self.receiver_buffer_in_use}")


        if self.track_states:
            with open('optimizer_call_level_states.csv', 'w') as f:
                f.write("Read Thread, Network Thread, Write Thread, Utility, Read Throughput, Sender Buffer, Network Throughput, Receiver Buffer, Write Throughput\n")

            with open('thread_level_states.csv', 'w') as f:
                f.write("Thread Type, Throughput, Sender Buffer, Receiver Buffer\n")
                f.write(f"Initial, 0, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")

    def read_thread_task(self, time):
        throughput_increase = 0
        if self.sender_buffer_in_use < self.sender_buffer_capacity:
            read_throughput_temp = min(self.read_throughput_per_thread, self.sender_buffer_capacity - self.sender_buffer_in_use)
            throughput_increase = min(read_throughput_temp, self.read_bandwidth-self.read_throughput)
            self.read_throughput += throughput_increase
            self.sender_buffer_in_use += throughput_increase

        time_taken = throughput_increase / self.read_throughput_per_thread
        next_time = time + time_taken + 0.001
        if next_time < 1:
            self.thread_queue.put((next_time, "read"))

        if throughput_increase > 0 and self.track_states:
            with open('thread_level_states.csv', 'a') as f:
                f.write(f"Read, {throughput_increase}, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")
        return next_time

    def network_thread_task(self, time):
        throughput_increase = 0
        # print(f"Network Thread start: Network Throughput: {throughput_increase}, Sender Buffer: {self.sender_buffer_in_use}, Receiver Buffer: {self.receiver_buffer_in_use}")
        if self.sender_buffer_in_use > 0 and self.receiver_buffer_in_use < self.receiver_buffer_capacity:
            network_throughput_temp = min(self.network_throughput_per_thread, self.sender_buffer_in_use, self.receiver_buffer_capacity - self.receiver_buffer_in_use)
            throughput_increase = min(network_throughput_temp, self.network_bandwidth-self.network_throughput)
            self.network_throughput += throughput_increase
            self.sender_buffer_in_use -= throughput_increase
            self.receiver_buffer_in_use += throughput_increase

        time_taken = throughput_increase / self.network_throughput_per_thread
        next_time = time + time_taken + 0.001
        if next_time < 1:
            self.thread_queue.put((next_time, "network"))
        # print(f"Network Thread end: Network Throughput: {throughput_increase}, Sender Buffer: {self.sender_buffer_in_use}, Receiver Buffer: {self.receiver_buffer_in_use}")
        if throughput_increase > 0 and self.track_states:
            with open('thread_level_states.csv', 'a') as f:
                f.write(f"Network, {throughput_increase}, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")
        return next_time

    def write_thread_task(self, time):
        throughput_increase = 0
        if self.receiver_buffer_in_use > 0:
            write_throughput_temp = min(self.write_throughput_per_thread, self.receiver_buffer_in_use)
            throughput_increase = min(write_throughput_temp, self.write_bandwidth-self.write_throughput)
            self.write_throughput += throughput_increase
            self.receiver_buffer_in_use -= throughput_increase

        time_taken = throughput_increase / self.write_throughput_per_thread
        next_time = time + time_taken + 0.001
        if next_time < 1:
            self.thread_queue.put((next_time, "write"))
        # print(f"Write Thread: Sender Buffer: {self.sender_buffer_in_use}, Receiver Buffer: {self.receiver_buffer_in_use}")
        if throughput_increase > 0 and self.track_states:
            with open('thread_level_states.csv', 'a') as f:
                f.write(f"Write, {throughput_increase}, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")
        return next_time

    def get_utility_value_dummy(self, threads):
        x1, x2, x3 = map(int, threads)
        return ((x1 - 1) ** 2 + (x2 - 2) ** 2 + (x3 + 3) ** 2 + \
            np.sin(2 * x1) + np.sin(2 * x2) + np.cos(2 * x3)) * -1

    def get_utility_value(self, threads):
        read_thread, network_thread, write_thread = map(int, threads)
        self.read_thread = read_thread
        self.network_thread = network_thread
        self.write_thread = write_thread

        self.thread_queue = PriorityQueue() # Key: time, Value: thread_type
        self.read_throughput = 0
        self.network_throughput = 0
        self.write_throughput = 0

        # populate the thread queue
        for i in range(read_thread):
            self.thread_queue.put((0, "read"))
        for i in range(network_thread):
            self.thread_queue.put((0, "network"))
        for i in range(write_thread):
            self.thread_queue.put((0, "write"))

        read_thread_finish_time = 0
        network_thread_finish_time = 0
        write_thread_finish_time = 0

        while not self.thread_queue.empty():
            time, thread_type = self.thread_queue.get()
            if thread_type == "read":
                read_thread_finish_time = self.read_thread_task(time)
            elif thread_type == "network":
                network_thread_finish_time = self.network_thread_task(time)
            elif thread_type == "write":
                write_thread_finish_time = self.write_thread_task(time)

        self.read_throughput = self.read_throughput / read_thread_finish_time
        self.network_throughput = self.network_throughput / network_thread_finish_time
        self.write_throughput = self.write_throughput / write_thread_finish_time

        self.sender_buffer_in_use = max(self.sender_buffer_in_use, 0)
        self.receiver_buffer_in_use = max(self.receiver_buffer_in_use, 0)

        utility = (self.read_throughput/self.K ** read_thread) + (self.network_throughput/self.K ** network_thread) + (self.write_throughput/self.K ** write_thread)

        # print(f"Read thread: {read_thread}, Network thread: {network_thread}, Write thread: {write_thread}, Utility: {utility}")

        if self.track_states:
            with open('optimizer_call_level_states.csv', 'a') as f:
                f.write(f"{read_thread}, {network_thread}, {write_thread}, {utility}, {self.read_throughput}, {self.sender_buffer_in_use}, {self.network_throughput}, {self.receiver_buffer_in_use}, {self.write_throughput}\n")

        final_state = SimulatorState(self.sender_buffer_capacity-self.sender_buffer_in_use,
                                     self.receiver_buffer_capacity-self.receiver_buffer_in_use,
                                     self.read_throughput, self.write_throughput, self.network_throughput,
                                     read_thread, write_thread, network_thread)

        return utility, final_state

class NetworkOptimizationEnv(gym.Env):
    def __init__(self, simulator=None):
        super(NetworkOptimizationEnv, self).__init__()
        oneGB = 1024
        self.simulator = NetworkSystemSimulator(sender_buffer_capacity=5*oneGB,
                                                receiver_buffer_capacity=3*oneGB,
                                                read_throughput_per_thread=100,
                                                network_throughput_per_thread=75,
                                                write_throughput_per_thread=35,
                                                read_bandwidth=6*oneGB,
                                                write_bandwidth=700,
                                                network_bandwidth=1*oneGB)
        if simulator is not None:
            self.simulator = simulator
        self.thread_limits = [1, 100]  # Threads can be between 1 and 10

        # Continuous action space: adjustments between -5.0 and +5.0
        self.action_space = spaces.Box(low=np.array([self.thread_limits[0]] * 3),
                               high=np.array([self.thread_limits[1]] * 3),
                               dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, self.thread_limits[0], self.thread_limits[0], self.thread_limits[0]]),
            high=np.array([
                self.simulator.sender_buffer_capacity,
                self.simulator.receiver_buffer_capacity,
                np.inf,  # Or maximum possible throughput values
                np.inf,
                np.inf,
                self.thread_limits[1],
                self.thread_limits[1],
                self.thread_limits[1]
            ]),
            dtype=np.float32
        )

        self.state = SimulatorState(sender_buffer_remaining_capacity=5*oneGB,
                                    receiver_buffer_remaining_capacity=3*oneGB,
                                    read_thread=1,
                                    network_thread=1,
                                    write_thread=1)
        self.max_steps = 100
        self.current_step = 0

        # For recording the trajectory
        self.trajectory = []

    def step(self, action):
        new_thread_counts = np.clip(np.round(action), self.thread_limits[0], self.thread_limits[1]).astype(np.int32)

        # Compute utility and update state
        utility, self.state = self.simulator.get_utility_value(new_thread_counts)

        # Penalize actions that hit thread limits
        penalty = 0
        if new_thread_counts[0] == self.thread_limits[0] or new_thread_counts[0] == self.thread_limits[1]:
            penalty -= 100  # Adjust penalty value as needed
        if new_thread_counts[1] == self.thread_limits[0] or new_thread_counts[1] == self.thread_limits[1]:
            penalty -= 100
        if new_thread_counts[2] == self.thread_limits[0] or new_thread_counts[2] == self.thread_limits[1]:
            penalty -= 100

        # Adjust reward
        reward = utility + penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Record the state
        self.trajectory.append(self.state.copy())

        # Return state as NumPy array
        return self.state.to_array(), reward, done, {}

    def reset(self):
        oneGB = 1024

        initial_read_thread = np.random.randint(self.thread_limits[0], self.thread_limits[1] + 1)
        initial_network_thread = np.random.randint(self.thread_limits[0], self.thread_limits[1] + 1)
        initial_write_thread = np.random.randint(self.thread_limits[0], self.thread_limits[1] + 1)

        self.state = SimulatorState(sender_buffer_remaining_capacity=5*oneGB,
                                    receiver_buffer_remaining_capacity=3*oneGB,
                                    read_thread=initial_read_thread,
                                    network_thread=initial_network_thread,
                                    write_thread=initial_write_thread)
        self.current_step = 0
        self.simulator.sender_buffer_in_use = 0
        self.simulator.receiver_buffer_in_use = 0
        self.trajectory = [self.state.copy()]

        # Return initial state as NumPy array
        return self.state.to_array()

class RecurrentPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(RecurrentPolicyNetwork, self).__init__()
        self.hidden_size = hidden_size
        # Project input state to hidden size
        self.fc = nn.Linear(state_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = -20
        self.log_std_max = 2

        self.to(device)

    def forward(self, state, hidden):
        """
        state: (batch_size, seq_len, state_dim)
        hidden: (h, c) each (1, batch_size, hidden_size)
        """
        x = torch.tanh(self.fc(state))
        x, (h, c) = self.lstm(x, hidden)
        # Take the last time step's output
        last_output = x[:, -1, :]
        mean = self.mean_layer(last_output)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std, (h, c)

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h, c)


class RecurrentValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(RecurrentValueNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(state_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.value_head = nn.Linear(hidden_size, 1)
        self.to(device)

    def forward(self, state, hidden):
        # state: (batch_size, seq_len, state_dim)
        x = torch.tanh(self.fc(state))
        x, (h, c) = self.lstm(x, hidden)
        last_output = x[:, -1, :]
        value = self.value_head(last_output)
        return value, (h, c)

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h, c)

class RecurrentPPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.policy = RecurrentPolicyNetwork(state_dim, action_dim)
        self.policy_old = RecurrentPolicyNetwork(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.value_function = RecurrentValueNetwork(state_dim)
        
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_function.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, hidden):
        # state should be (state_dim,) for a single step
        # Add batch and seq_len dimensions: (1,1,state_dim)
        state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            mean, std, hidden = self.policy_old(state, hidden)
            dist = Normal(mean, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.squeeze(0).cpu().numpy(), action_logprob.cpu().numpy(), hidden

    def update(self, memory):
        # PPO updates are more complex with LSTM:
        # Normally, you'd process entire sequences at once.
        # This code snippet is a placeholder. You need to:
        # 1. Create batches of sequences from memory.
        # 2. Compute advantages and returns per sequence.
        # 3. Run forward pass through policy and value networks in sequence.
        
        states = torch.tensor(memory.states, dtype=torch.float32).to(device)
        actions = torch.tensor(memory.actions, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)

        # For simplicity, assume one big sequence:
        states = states.unsqueeze(0) # (1, T, state_dim)
        actions = actions.unsqueeze(0) # (1, T, action_dim)
        old_logprobs = old_logprobs.unsqueeze(0) # (1, T, action_dim)
        rewards = rewards.unsqueeze(0) # (1, T)

        # Compute returns:
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards.squeeze(0).cpu().numpy()):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(0).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Evaluate actions:
        h = self.policy.init_hidden(batch_size=1)
        mean, std, _ = self.policy(states, h)
        dist = Normal(mean, std)
        logprobs = dist.log_prob(actions)
        logprobs = logprobs.sum(dim=-1)  # sum across action dims if needed

        h_v = self.value_function.init_hidden(batch_size=1)
        values, _ = self.value_function(states, h_v)
        values = values.squeeze(-1)

        advantages = returns - values.detach()

        # PPO loss:
        logprobs_old = old_logprobs.sum(dim=-1)
        ratios = torch.exp(logprobs - logprobs_old)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2)

        value_loss = self.MseLoss(values, returns)
        loss = policy_loss.mean() + 0.5 * value_loss  # add entropy if needed

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.hiddens = []  # Store hidden states if needed

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.hiddens[:]


from tqdm import tqdm

def train_ppo(env, agent, max_episodes=1000):
    memory = Memory()
    total_rewards = []
    for episode in tqdm(range(1, max_episodes + 1), desc="Episodes"):
        state = env.reset()
        episode_reward = 0
        # Initialize hidden states for policy and value
        policy_hidden = agent.policy_old.init_hidden()
        
        for t in range(env.max_steps):
            action, action_logprob, policy_hidden = agent.select_action(state, policy_hidden)
            next_state, reward, done, _ = env.step(action)
            
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)
            
            state = next_state
            episode_reward += reward
            if done:
                break

        agent.update(memory)

        with open('episode_rewards_recurrent.csv', 'a') as f:
                f.write(f"Episode {episode}, Last State: {np.round(state[-3:])}, Reward: {reward}\n")

        memory.clear()
        total_rewards.append(episode_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
        if episode % 1000 == 0:
            save_model(agent, "models/recurrent_policy_"+ str(episode) +".pth", "models/recurrent_value_"+ str(episode) +".pth")
            print("Model saved successfully.")

    return total_rewards


def plot_rewards(rewards, title, pdf_file):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.grid(True)
    
    plt.savefig(pdf_file)  
    plt.close()

import os
import re

def find_last_policy_model():
    models = os.listdir("models")
    models = [model for model in models if re.match(r'recurrent_policy_\d+\.pth', model)]
    models.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return models[-1]

def find_last_value_model():
    models = os.listdir("models")
    models = [model for model in models if re.match(r'recurrent_value_\d+\.pth', model)]
    models.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return models[-1]

if __name__ == '__main__':
    env = NetworkOptimizationEnv()
    agent = RecurrentPPOAgent(state_dim=8, action_dim=3, lr=1e-4, eps_clip=0.1)
    print("Training PPO agent on Network System Simulator with continuous actions...")
    rewards = train_ppo(env, agent, max_episodes=50000)

    plot_rewards(rewards, 'PPO Training Rewards', 'training_rewards_recurrent.pdf')

    env = NetworkOptimizationEnv()
    agent = RecurrentPPOAgent(state_dim=8, action_dim=3, lr=1e-4, eps_clip=0.1)

    policy_model = find_last_policy_model()
    value_model = find_last_value_model()

    print(f"Loading model... Value: {value_model}, Policy: {policy_model}")
    load_model(agent, "models/"+policy_model, "models/"+value_model)
    print("Model loaded successfully.")
    rewards = train_ppo(env, agent, max_episodes=1000)

    plot_rewards(rewards, 'PPO Inference Rewards', 'inference_rewards_recurrent.pdf')