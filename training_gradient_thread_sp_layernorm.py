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
    def __init__(
        self,
        sender_buffer_remaining_capacity=0,
        receiver_buffer_remaining_capacity=0,
        read_throughput_change=0,
        write_throughput_change=0,
        network_throughput_change=0,
        read_thread_change=0,
        write_thread_change=0,
        network_thread_change=0,
        read_thread=0,
        write_thread=0,
        network_thread=0,
        rewards_change=0
    ):
        self.sender_buffer_remaining_capacity = sender_buffer_remaining_capacity
        self.receiver_buffer_remaining_capacity = receiver_buffer_remaining_capacity
        self.read_throughput_change = read_throughput_change
        self.write_throughput_change = write_throughput_change
        self.network_throughput_change = network_throughput_change
        self.read_thread_change = read_thread_change
        self.write_thread_change = write_thread_change
        self.network_thread_change = network_thread_change
        self.read_thread = read_thread
        self.write_thread = write_thread
        self.network_thread = network_thread
        self.rewards_change = rewards_change

    def copy(self):
        return SimulatorState(
            sender_buffer_remaining_capacity=self.sender_buffer_remaining_capacity,
            receiver_buffer_remaining_capacity=self.receiver_buffer_remaining_capacity,
            read_throughput_change=self.read_throughput_change,
            write_throughput_change=self.write_throughput_change,
            network_throughput_change=self.network_throughput_change,
            read_thread_change=self.read_thread_change,
            write_thread_change=self.write_thread_change,
            network_thread_change=self.network_thread_change,
            read_thread=self.read_thread,
            write_thread=self.write_thread,
            network_thread=self.network_thread,
            rewards_change=self.rewards_change
        )

    def to_array(self):
        return np.array([
            self.sender_buffer_remaining_capacity,
            self.receiver_buffer_remaining_capacity,
            self.read_throughput_change,
            self.write_throughput_change,
            self.network_throughput_change,
            self.read_thread_change,
            self.write_thread_change,
            self.network_thread_change,
            self.read_thread,
            self.write_thread,
            self.network_thread,
            self.rewards_change
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

        min_bandwidth = min(read_bandwidth, write_bandwidth, network_bandwidth)

        optimal_read_thread = math.ceil(min_bandwidth // read_throughput_per_thread)
        optimal_network_thread = math.ceil(min_bandwidth // network_throughput_per_thread)
        optimal_write_thread = math.ceil(min_bandwidth // write_throughput_per_thread)

        self.optimal_reward_read = (min_bandwidth/self.K ** optimal_read_thread)
        self.optimal_reward_network = (min_bandwidth/self.K ** optimal_network_thread)
        self.optimal_reward_write = (min_bandwidth/self.K ** optimal_write_thread)
        
        self.reward = 0
        self.prev_read_throughput = 0
        self.prev_network_throughput = 0
        self.prev_write_throughput = 0

        # Initialize the buffers
        self.sender_buffer_in_use = max(min(self.read_throughput_per_thread * read_thread - self.network_throughput_per_thread * self.network_thread, self.sender_buffer_capacity), 0)
        self.receiver_buffer_in_use = max(min(self.network_throughput_per_thread * network_thread - self.write_throughput_per_thread * self.write_thread, self.receiver_buffer_capacity), 0)

        print(f"Initial Sender Buffer: {self.sender_buffer_in_use}, Receiver Buffer: {self.receiver_buffer_in_use}")


        # if self.track_states:
        #     with open('optimizer_call_level_states.csv', 'w') as f:
        #         f.write("Read Thread, Network Thread, Write Thread, Utility, Read Throughput, Sender Buffer, Network Throughput, Receiver Buffer, Write Throughput\n")

        #     with open('thread_level_states.csv', 'w') as f:
        #         f.write("Thread Type, Throughput, Sender Buffer, Receiver Buffer\n")
        #         f.write(f"Initial, 0, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")

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

        # if throughput_increase > 0 and self.track_states:
        #     with open('thread_level_states.csv', 'a') as f:
        #         f.write(f"Read, {throughput_increase}, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")
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
        # if throughput_increase > 0 and self.track_states:
        #     with open('thread_level_states.csv', 'a') as f:
        #         f.write(f"Network, {throughput_increase}, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")
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
        # if throughput_increase > 0 and self.track_states:
        #     with open('thread_level_states.csv', 'a') as f:
        #         f.write(f"Write, {throughput_increase}, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")
        return next_time

    def get_utility_value_dummy(self, threads):
        x1, x2, x3 = map(int, threads)
        return ((x1 - 1) ** 2 + (x2 - 2) ** 2 + (x3 + 3) ** 2 + \
            np.sin(2 * x1) + np.sin(2 * x2) + np.cos(2 * x3)) * -1

    def get_utility_value(self, threads):
        read_thread, network_thread, write_thread = map(int, threads)

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

        utility_read = (self.read_throughput/self.K ** read_thread)
        utility_network = (self.network_throughput/self.K ** network_thread)
        utility_write = (self.write_throughput/self.K ** write_thread)

        # print(f"Read thread: {read_thread}, Network thread: {network_thread}, Write thread: {write_thread}, Utility: {utility}")

        if self.track_states:
            with open('threads_training_gradient_thread_sp_layernorm.csv', 'a') as f:
                f.write(f"{read_thread}, {network_thread}, {write_thread}\n")
            with open('throughputs_training_gradient_thread_sp_layernorm.csv', 'a') as f:
                f.write(f"{self.read_throughput}, {self.network_throughput}, {self.write_throughput}\n")

        reward = (((utility_read/self.optimal_reward_read)
                  *(utility_network/self.optimal_reward_network)
                  *(utility_write/self.optimal_reward_write))**(1/3)) * 100
        
        final_state = SimulatorState((self.sender_buffer_capacity-self.sender_buffer_in_use)/self.sender_buffer_capacity,
                                    (self.receiver_buffer_capacity-self.receiver_buffer_in_use)/self.receiver_buffer_capacity,
                                    (self.read_throughput - self.prev_read_throughput)/self.prev_read_throughput if self.prev_read_throughput > 0 else 0,
                                    (self.write_throughput - self.prev_write_throughput)/self.prev_write_throughput if self.prev_write_throughput > 0 else 0,
                                    (self.network_throughput - self.prev_network_throughput)/self.prev_network_throughput if self.prev_network_throughput > 0 else 0,
                                    (read_thread - self.read_thread)/self.read_thread,
                                    (write_thread - self.write_thread)/self.write_thread,
                                    (network_thread - self.network_thread)/self.network_thread,
                                    read_thread,
                                    write_thread,
                                    network_thread,
                                    (reward-self.reward)/self.reward if self.reward > 0 else 0
                                    )                                    
        


        self.read_thread = read_thread
        self.network_thread = network_thread
        self.write_thread = write_thread
        self.reward = reward
        self.prev_read_throughput = self.read_throughput
        self.prev_network_throughput = self.network_throughput
        self.prev_write_throughput = self.write_throughput

        return reward, final_state

import math
class SimulatorGenerator:
    def generate_simulator(self):
        oneGB = 1024
        sender_buffer_capacity = max(1, np.random.poisson(lam=50)) * oneGB
        receiver_buffer_capacity = max(1, np.random.poisson(lam=50)) * oneGB
        read_throughput_per_thread = max(1, np.random.poisson(lam=1000))
        network_throughput_per_thread = max(1, np.random.poisson(lam=1000))
        write_throughput_per_thread = max(1, np.random.poisson(lam=1000))
        read_bandwidth = max(1, np.random.poisson(lam=12)) * oneGB
        write_bandwidth = max(1, np.random.poisson(lam=12)) * oneGB
        network_bandwidth = max(1, np.random.poisson(lam=12)) * oneGB

        simulator = NetworkSystemSimulator(sender_buffer_capacity=sender_buffer_capacity,
                                            receiver_buffer_capacity=receiver_buffer_capacity,
                                            read_throughput_per_thread=read_throughput_per_thread,
                                            network_throughput_per_thread=network_throughput_per_thread,
                                            write_throughput_per_thread=write_throughput_per_thread,
                                            read_bandwidth=read_bandwidth,
                                            write_bandwidth=write_bandwidth,
                                            network_bandwidth=network_bandwidth,
                                            track_states=True)

        min_bandwidth = min(read_bandwidth, write_bandwidth, network_bandwidth)

        optimal_read_thread = math.ceil(min_bandwidth // read_throughput_per_thread)
        optimal_network_thread = math.ceil(min_bandwidth // network_throughput_per_thread)
        optimal_write_thread = math.ceil(min_bandwidth // write_throughput_per_thread)

        optimals = [optimal_read_thread, optimal_network_thread, optimal_write_thread, min_bandwidth]
        
        return optimals, simulator

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
                                                network_bandwidth=1*oneGB,
                                                track_states=True)
        if simulator is not None:
            self.simulator = simulator
        self.thread_limits = [1, 100]  # Threads can be between 1 and 10

        # Continuous action space: adjustments between -5.0 and +5.0
        self.action_space = spaces.Box(low=np.array([self.thread_limits[0]] * 3),
                               high=np.array([self.thread_limits[1]] * 3),
                               dtype=np.float32)

        read_bw = self.simulator.read_bandwidth
        write_bw = self.simulator.write_bandwidth
        network_bw = self.simulator.network_bandwidth
        sb_capacity = self.simulator.sender_buffer_capacity
        rb_capacity = self.simulator.receiver_buffer_capacity
        thread_delta_max = self.thread_limits[1] - self.thread_limits[0]
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,
                0.0,
                -read_bw,
                -write_bw,
                -network_bw,
                -thread_delta_max,
                -thread_delta_max,
                -thread_delta_max,
                float(self.thread_limits[0]),
                float(self.thread_limits[0]),
                float(self.thread_limits[0]),
                -1.0
            ], dtype=np.float32),
            high=np.array([
                float(sb_capacity),
                float(rb_capacity),
                float(read_bw),
                float(write_bw),
                float(network_bw),
                float(thread_delta_max),
                float(thread_delta_max),
                float(thread_delta_max),
                float(self.thread_limits[1]),
                float(self.thread_limits[1]),
                float(self.thread_limits[1]),
                1.0
            ], dtype=np.float32),
            dtype=np.float32
        )


        self.state = SimulatorState(sender_buffer_remaining_capacity=self.simulator.sender_buffer_capacity,
                                    receiver_buffer_remaining_capacity=self.simulator.receiver_buffer_capacity)
        self.max_steps = 5
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
            penalty -= 10  # Adjust penalty value as needed
        if new_thread_counts[1] == self.thread_limits[0] or new_thread_counts[1] == self.thread_limits[1]:
            penalty -= 10
        if new_thread_counts[2] == self.thread_limits[0] or new_thread_counts[2] == self.thread_limits[1]:
            penalty -= 10

        # Adjust reward
        reward = utility + penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Record the state
        self.trajectory.append(self.state.copy())

        # Return state as NumPy array
        return self.state.to_array(), reward, done, {}

    def reset(self, simulator=None):
        if simulator is not None:
            self.simulator = simulator
            self.state = SimulatorState(sender_buffer_remaining_capacity=self.simulator.sender_buffer_capacity,
                                    receiver_buffer_remaining_capacity=self.simulator.receiver_buffer_capacity)
            read_bw = self.simulator.read_bandwidth
            write_bw = self.simulator.write_bandwidth
            network_bw = self.simulator.network_bandwidth
            thread_delta_max = self.thread_limits[1] - self.thread_limits[0]
            self.observation_space = spaces.Box(
                low=np.array([
                    0.0,
                    0.0,
                    -read_bw,
                    -write_bw,
                    -network_bw,
                    -thread_delta_max,
                    -thread_delta_max,
                    -thread_delta_max,
                    float(self.thread_limits[0]),
                    float(self.thread_limits[0]),
                    float(self.thread_limits[0]),
                    -np.inf
                ], dtype=np.float32),
                high=np.array([
                    1.0,
                    1.0,
                    float(read_bw),
                    float(write_bw),
                    float(network_bw),
                    float(thread_delta_max),
                    float(thread_delta_max),
                    float(thread_delta_max),
                    float(self.thread_limits[1]),
                    float(self.thread_limits[1]),
                    float(self.thread_limits[1]),
                    np.inf
                ], dtype=np.float32),
                dtype=np.float32
            )


        self.current_step = 0
        self.trajectory = [self.state.copy()]

        # Return initial state as NumPy array
        return self.state.to_array()

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, size, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.ln1 = nn.LayerNorm(size)
        self.fc2 = nn.Linear(size, size)
        self.ln2 = nn.LayerNorm(size)
        self.activation = activation()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out += residual
        out = self.activation(out)
        return out

class PolicyNetworkContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetworkContinuous, self).__init__()
        self.input_layer = nn.Linear(state_dim, 256)
        self.ln_in = nn.LayerNorm(256)

        # Optional: Use residual blocks that include LayerNorm internally
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256, activation=nn.ReLU) for _ in range(3)
        ])

        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.to(device)
        
    def forward(self, state):
        x = self.input_layer(state)
        x = self.ln_in(x)
        x = torch.tanh(x)

        for block in self.res_blocks:
            x = block(x)

        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc_in = nn.Linear(state_dim, 256)
        self.ln_in = nn.LayerNorm(256)

        # Two residual blocks with LayerNorm inside
        self.res_block1 = ResidualBlock(256, activation=nn.Tanh)
        self.res_block2 = ResidualBlock(256, activation=nn.Tanh)

        self.fc_out = nn.Linear(256, 1)
        self.to(device)

    def forward(self, state):
        x = self.fc_in(state)
        x = self.ln_in(x)
        x = torch.tanh(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        value = self.fc_out(x)
        return value

class PPOAgentContinuous:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNetworkContinuous(state_dim, action_dim)
        self.policy_old = PolicyNetworkContinuous(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_function = ValueNetwork(state_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_function.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, is_inference=False, std_scale=0.1):
        state = torch.FloatTensor(state).to(device)
        mean, std = self.policy_old(state)
        if is_inference:
            reduced_std = std_scale * std  # or some small fraction
            dist = Normal(mean, reduced_std)
        else:
            # During training, sample from the distribution
            dist = Normal(mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy()

    def update(self, memory):
        states = torch.stack(memory.states).to(device)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32).to(device)

        # Compute discounted rewards
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Get new action probabilities
        mean, std = self.policy(states)
        dist = Normal(mean, std)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        logprobs = logprobs.sum(dim=1)
        old_logprobs = old_logprobs.sum(dim=1)
        entropy = entropy.sum(dim=1)

        ratios = torch.exp(logprobs - old_logprobs)
        state_values = self.value_function(states).squeeze()

        # Compute advantage
        advantages = returns - state_values.detach()

        # Surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.1 * entropy

        # Update policy
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]


from tqdm import tqdm

def train_ppo(env, agent, max_episodes=1000, is_inference=False, std_scale=0.1):
    memory = Memory()
    total_rewards = []
    for episode in tqdm(range(1, max_episodes + 1), desc="Episodes"):
        state = None
        simulator_generator = SimulatorGenerator()
        if episode % 25000 == 0:
            _, simulator = simulator_generator.generate_simulator()
            state = env.reset(simulator=simulator)
        else:
            state = env.reset()
        episode_reward = 0
        for t in range(env.max_steps):
            action, action_logprob = agent.select_action(state, is_inference=is_inference, std_scale=std_scale)
            next_state, reward, done, _ = env.step(action)

            memory.states.append(torch.FloatTensor(state).to(device))
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)

            state = next_state
            if t==0:
                episode_reward += reward
            if done:
                break

        agent.update(memory)

        # print(f"Episode {episode}\tLast State: {state}\tReward: {reward}")
        with open('episode_rewards_training_gradient_thread_sp_leyernorm.csv', 'a') as f:
                f.write(f"Episode {episode}, Last State: {np.round(state[-3:])}, Reward: {reward}\n")

        memory.clear()
        total_rewards.append(episode_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
        if episode % 1000 == 0:
            save_model(agent, "models/training_gradient_thread_sp_leyernorm_policy_"+ str(episode) +".pth", "models/training_gradient_thread_sp_leyernorm_value_"+ str(episode) +".pth")
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

import csv

import pandas as pd

def plot_threads_csv(threads_file='threads_training_gradient_thread_sp_layernorm.csv', optimals = None, output_file='threads_plot.png'):
    optimal_read, optimal_network, optimal_write, _ = optimals
    data = []

    # Read data from threads_training_gradient_thread_sp_layernorm.csv
    with open(threads_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            data.append([float(value) for value in row[:3]])

    df = pd.DataFrame(data, columns=['Read Threads', 'Network Threads', 'Write Threads'])

    # Compute rolling averages
    rolling_read = df['Read Threads'].rolling(window=5).mean()
    rolling_network = df['Network Threads'].rolling(window=5).mean()
    rolling_write = df['Write Threads'].rolling(window=5).mean()

    # Create subplots for each type
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(rolling_read, label='Read Threads (5-point MA)')
    plt.title('Read Threads (Stable: '+ str(optimal_read) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Thread Count')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(rolling_network, label='Network Threads (5-point MA)', color='orange')
    plt.title('Network Threads (Stable: '+ str(optimal_network) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Thread Count')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(rolling_write, label='Write Threads (5-point MA)', color='green')
    plt.title('Write Threads (Stable: '+ str(optimal_write) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Thread Count')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved thread count plot to {output_file}")

# Function to plot throughputs with rolling averages
def plot_throughputs_csv(throughputs_file='throughputs_training_gradient_thread_sp_layernorm.csv', optimals = None, output_file='throughputs_plot.png'):
    optimal_throughput = optimals[-1]
    data = []

    # Read data from throughputs_training_gradient_thread_sp_layernorm.csv
    with open(throughputs_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            data.append([float(value) for value in row[:3]])

    df = pd.DataFrame(data, columns=['Read Throughput', 'Network Throughput', 'Write Throughput'])

    # Compute rolling averages
    rolling_read = df['Read Throughput'].rolling(window=5).mean()
    rolling_network = df['Network Throughput'].rolling(window=5).mean()
    rolling_write = df['Write Throughput'].rolling(window=5).mean()

    # Create subplots for each type
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(rolling_read, label='Read Throughput')
    plt.title('Read Throughput (Stable: '+ str(optimal_throughput) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(rolling_network, label='Network Throughput', color='orange')
    plt.title('Network Throughput (Stable: '+ str(optimal_throughput) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(rolling_write, label='Write Throughput', color='green')
    plt.title('Write Throughput (Stable: '+ str(optimal_throughput) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved throughput plot to {output_file}")


import os
import re

def find_last_policy_model():
    models = os.listdir("models")
    models = [model for model in models if re.match(r'training_gradient_thread_sp_leyernorm_policy_\d+\.pth', model)]
    models.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return models[-1]

def find_last_value_model():
    models = os.listdir("models")
    models = [model for model in models if re.match(r'training_gradient_thread_sp_leyernorm_value_\d+\.pth', model)]
    models.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return models[-1]

if __name__ == '__main__':
    # if os.path.exists('threads_training_gradient_thread_sp_layernorm.csv'):
    #     os.remove('threads_training_gradient_thread_sp_layernorm.csv')
    # if os.path.exists('throughputs_training_gradient_thread_sp_layernorm.csv'):
    #     os.remove('throughputs_training_gradient_thread_sp_layernorm.csv')

    # oneGB = 1024
    # simulator = NetworkSystemSimulator(sender_buffer_capacity=10*oneGB,
    #                                             receiver_buffer_capacity=6*oneGB,
    #                                             read_throughput_per_thread=200,
    #                                             network_throughput_per_thread=150,
    #                                             write_throughput_per_thread=70,
    #                                             read_bandwidth=12*oneGB,
    #                                             write_bandwidth=2*oneGB,
    #                                             network_bandwidth=2*oneGB,
    #                                             track_states=True)
    # env = NetworkOptimizationEnv(simulator=simulator)
    # agent = PPOAgentContinuous(state_dim=12, action_dim=3, lr=1e-4, eps_clip=0.1)
    # rewards = train_ppo(env, agent, max_episodes=250000)
    
    # plot_rewards(rewards, 'PPO Training Rewards', 'training_rewards_training_gradient_thread_sp_leyernorm.pdf')

    inference_count = 5
    simulator_generator = SimulatorGenerator()
    for i in range(inference_count):
        if os.path.exists('threads_training_gradient_thread_sp_layernorm.csv'):
            os.remove('threads_training_gradient_thread_sp_layernorm.csv')
        if os.path.exists('throughputs_training_gradient_thread_sp_layernorm.csv'):
            os.remove('throughputs_training_gradient_thread_sp_layernorm.csv')

        optimals, simulator = simulator_generator.generate_simulator()

        # save simulator parameters to a file
        with open('simulators/simulator_parameters_'+ str(i) +'.csv', 'w') as f:
            f.write(f"Sender Buffer Capacity, {simulator.sender_buffer_capacity}\n")
            f.write(f"Receiver Buffer Capacity, {simulator.receiver_buffer_capacity}\n")
            f.write(f"Read Throughput per Thread, {simulator.read_throughput_per_thread}\n")
            f.write(f"Network Throughput per Thread, {simulator.network_throughput_per_thread}\n")
            f.write(f"Write Throughput per Thread, {simulator.write_throughput_per_thread}\n")
            f.write(f"Read Bandwidth, {simulator.read_bandwidth}\n")
            f.write(f"Write Bandwidth, {simulator.write_bandwidth}\n")
            f.write(f"Network Bandwidth, {simulator.network_bandwidth}\n")

        env = NetworkOptimizationEnv(simulator=simulator)
        agent = PPOAgentContinuous(state_dim=12, action_dim=3, lr=1e-4, eps_clip=0.1)

        policy_model = 'training_gradient_thread_sp_leyernorm_policy_250000.pth'
        value_model = 'training_gradient_thread_sp_leyernorm_value_250000.pth'

        print(f"Loading model... Value: {value_model}, Policy: {policy_model}")
        load_model(agent, "models/"+policy_model, "models/"+value_model)
        print("Model loaded successfully.")

        rewards = train_ppo(env, agent, max_episodes=20, is_inference=True, std_scale=0.1)

        plot_rewards(rewards, 'PPO Inference Rewards', 'rewards/inference_rewards_training_gradient_thread_sp_leyernorm_'+ str(i) +'.pdf')
        plot_threads_csv('threads_training_gradient_thread_sp_layernorm.csv', optimals, 'threads/inference_threads_plot_training_gradient_thread_sp_leyernorm_'+ str(i) +'.png')
        plot_throughputs_csv('throughputs_training_gradient_thread_sp_layernorm.csv', optimals, 'throughputs/inference_throughputs_plot_training_gradient_thread_sp_leyernorm_'+ str(i) +'.png') 