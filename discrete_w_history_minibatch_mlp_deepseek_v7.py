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

import copy
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
        read_thread=1,
        write_thread=1,
        network_thread=1,
        rewards_change=0,
        history_length=5  # Store last 5 states
    ):
        # Current state variables
        self.sender_buffer_remaining_capacity = sender_buffer_remaining_capacity
        self.receiver_buffer_remaining_capacity = receiver_buffer_remaining_capacity
        self.read_thread = read_thread
        self.write_thread = write_thread
        self.network_thread = network_thread
        
        # Historical data
        self.history_length = history_length
        self.throughput_history = {
            'read': [0] * (history_length-1) + [read_throughput_change],
            'write': [0] * (history_length-1) + [write_throughput_change],
            'network': [0] * (history_length-1) + [network_throughput_change]
        }
        self.thread_history = {
            'read': [0] * (history_length-1) + [read_thread_change],
            'write': [0] * (history_length-1) + [write_thread_change],
            'network': [0] * (history_length-1) + [network_thread_change]
        }
        self.reward_history = [0] * (history_length-1) + [rewards_change]

    def copy(self):
        return copy.deepcopy(self)

    def update_state(
            self,
            simulator_state = None
    ):
        if simulator_state is not None:
            self.sender_buffer_remaining_capacity = simulator_state.sender_buffer_remaining_capacity
            self.receiver_buffer_remaining_capacity = simulator_state.receiver_buffer_remaining_capacity
            self.read_thread = simulator_state.read_thread
            self.write_thread = simulator_state.write_thread
            self.network_thread = simulator_state.network_thread

            # Update historical data
            self.throughput_history['read'] = self.throughput_history['read'][1:] + [simulator_state.throughput_history['read'][-1]]
            self.throughput_history['write'] = self.throughput_history['write'][1:] + [simulator_state.throughput_history['write'][-1]]
            self.throughput_history['network'] = self.throughput_history['network'][1:] + [simulator_state.throughput_history['network'][-1]]
            self.thread_history['read'] = self.thread_history['read'][1:] + [simulator_state.thread_history['read'][-1]]
            self.thread_history['write'] = self.thread_history['write'][1:] + [simulator_state.thread_history['write'][-1]]
            self.thread_history['network'] = self.thread_history['network'][1:] + [simulator_state.thread_history['network'][-1]]
            self.reward_history = self.reward_history[1:] + [simulator_state.reward_history[-1]]

    def to_array(self):
        # Convert current state and history to a flat array
        current_state = np.array([
            self.sender_buffer_remaining_capacity,
            self.receiver_buffer_remaining_capacity,
            self.read_thread,
            self.write_thread,
            self.network_thread
        ], dtype=np.float32)
        
        # Add historical data
        history = np.concatenate([
            self.throughput_history['read'],
            self.throughput_history['write'],
            self.throughput_history['network'],
            self.thread_history['read'],
            self.thread_history['write'],
            self.thread_history['network'],
            self.reward_history
        ])
        
        return np.concatenate([current_state, history])
    
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
        self.K = 1.05

        min_bandwidth = min(read_bandwidth, write_bandwidth, network_bandwidth)

        self.optimal_read_thread = math.ceil(min_bandwidth // read_throughput_per_thread)
        self.optimal_network_thread = math.ceil(min_bandwidth // network_throughput_per_thread)
        self.optimal_write_thread = math.ceil(min_bandwidth // write_throughput_per_thread)

        self.optimal_reward_read = (min_bandwidth/self.K ** self.optimal_read_thread)
        self.optimal_reward_network = (min_bandwidth/self.K ** self.optimal_network_thread)
        self.optimal_reward_write = (min_bandwidth/self.K ** self.optimal_write_thread)
        
        self.reward = 0
        self.prev_read_throughput = 0
        self.prev_network_throughput = 0
        self.prev_write_throughput = 0

        self.utility_read = 0
        self.utility_network = 0
        self.utility_write = 0

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
        next_time = time + time_taken + 0.01
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
        next_time = time + time_taken + 0.01
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
        next_time = time + time_taken + 0.01
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
            with open('threads_dicrete_w_history_minibatch_mlp_deepseek_v7.csv', 'a') as f:
                f.write(f"{read_thread}, {network_thread}, {write_thread}\n")
            with open('throughputs_dicrete_w_history_minibatch_mlp_deepseek_v7.csv', 'a') as f:
                f.write(f"{self.read_throughput}, {self.network_throughput}, {self.write_throughput}\n")

        throughput_reward = ((self.read_throughput + self.network_throughput + self.write_throughput)/3) / min(self.read_bandwidth, self.write_bandwidth, self.network_bandwidth)
        thread_penalty = (read_thread/self.optimal_read_thread - 1)**2 + (network_thread/self.optimal_network_thread - 1)**2 + (write_thread/self.optimal_write_thread - 1)**2
        reward = throughput_reward - 0.3 * thread_penalty
        
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

        read_grad = (utility_read-self.utility_read)/(read_thread-self.read_thread) if (read_thread-self.read_thread) > 0 else 0
        network_grad = (utility_network-self.utility_network)/(network_thread-self.network_thread) if (network_thread-self.network_thread) > 0 else 0
        write_grad = (utility_write-self.utility_write)/(write_thread-self.write_thread) if (write_thread-self.write_thread) > 0 else 0
        grads = [read_grad, network_grad, write_grad]
        grads = np.array(grads, dtype=np.float32)                                    

        self.utility_read = utility_read
        self.utility_network = utility_network
        self.utility_write = utility_write
        self.read_thread = read_thread
        self.network_thread = network_thread
        self.write_thread = write_thread
        self.reward = reward
        self.prev_read_throughput = self.read_throughput
        self.prev_network_throughput = self.network_throughput
        self.prev_write_throughput = self.write_throughput

        throughputs = [self.read_throughput, 
                      self.network_throughput,
                      self.write_throughput]
        bottleneck_idx = np.argmin(throughputs)

        return reward, final_state, grads, bottleneck_idx

import math
class SimulatorGenerator:
    def generate_simulator(self, episode=1):
        factor = max(4 - (episode/100000), 1)

        oneGB = 1024
        sender_buffer_capacity = max(15, int(np.random.normal(loc=50, scale=10/factor))) * oneGB
        receiver_buffer_capacity = max(15, int(np.random.normal(loc=50, scale=10/factor))) * oneGB
        
        read_throughput_per_thread = max(300, int(np.random.normal(loc=1000, scale=300/factor)))
        network_throughput_per_thread = max(300, int(np.random.normal(loc=1000, scale=300/factor)))
        write_throughput_per_thread = max(300, int(np.random.normal(loc=1000, scale=300/factor)))
        
        read_bandwidth = max(4, int(np.random.normal(loc=12, scale=3/factor))) * oneGB
        write_bandwidth = max(4, int(np.random.normal(loc=12, scale=3/factor))) * oneGB
        network_bandwidth = max(4, int(np.random.normal(loc=12, scale=3/factor))) * oneGB

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
    def __init__(self, simulator=None, history_length=5):
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

        self.action_space = spaces.MultiDiscrete([5, 5, 5])
        obs_dim = 5 + 7 * history_length
        
        # Define an unbounded Box of shape (obs_dim,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.history_length = history_length

        self.state = SimulatorState(
            sender_buffer_remaining_capacity=self.simulator.sender_buffer_capacity,
            receiver_buffer_remaining_capacity=self.simulator.receiver_buffer_capacity,
            history_length=history_length
        )
        self.max_steps = 5
        self.current_step = 0

        # For recording the trajectory
        self.trajectory = []

    def step(self, action):
        deltas_map = [-3, -1, 0, +1, +3]
        
        read_index, net_index, write_index = action
        # Convert those indexes to actual deltas:
        read_delta = deltas_map[read_index]
        net_delta = deltas_map[net_index]
        write_delta = deltas_map[write_index]

        # 2) Compute new thread counts
        new_read = min(max(self.simulator.read_thread + read_delta, self.thread_limits[0]), self.thread_limits[1])
        new_network = min(max(self.simulator.network_thread + net_delta, self.thread_limits[0]), self.thread_limits[1])
        new_write = min(max(self.simulator.write_thread + write_delta, self.thread_limits[0]), self.thread_limits[1])
        new_thread_counts = [new_read, new_network, new_write]

        # Compute utility and update state
        utility, new_state, grads, bottleneck_idx = self.simulator.get_utility_value(new_thread_counts)
        self.state.update_state(new_state)
        # print(f"ACTION: {action}")
        # print(f"New Thread Counts: {new_thread_counts}")

        # Penalize actions that hit thread limits
        penalty = 0
        if new_thread_counts[0] == self.thread_limits[0] or new_thread_counts[0] == self.thread_limits[1]:
            penalty -= 0.60  # Adjust penalty value as needed
        if new_thread_counts[1] == self.thread_limits[0] or new_thread_counts[1] == self.thread_limits[1]:
            penalty -= 0.60
        if new_thread_counts[2] == self.thread_limits[0] or new_thread_counts[2] == self.thread_limits[1]:
            penalty -= 0.60

        # Add penalty for large changes
        # change_penalty = -0.1 * np.sum(np.abs(action)) / self.max_delta
        change_penalty = 0
        
        # Adjust reward
        reward = utility + penalty + change_penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Record the state
        self.trajectory.append(self.state.copy())

        # Return state as NumPy array
        return self.state.to_array(), reward, grads, bottleneck_idx, done, {}

    def reset(self, simulator=None):
        if simulator is not None:
            self.simulator = simulator
            
        self.simulator.sender_buffer_in_use = self.simulator.sender_buffer_capacity
        self.simulator.receiver_buffer_in_use = self.simulator.receiver_buffer_capacity
        self.simulator.read_thread = np.random.randint(1, 20)
        self.simulator.network_thread = np.random.randint(1, 20)
        self.simulator.write_thread = np.random.randint(1, 20)
        self.simulator.prev_read_throughput = 0
        self.simulator.prev_network_throughput = 0
        self.simulator.prev_write_throughput = 0
        self.simulator.reward = 0
        self.state = SimulatorState(
            sender_buffer_remaining_capacity=self.simulator.sender_buffer_capacity,
            receiver_buffer_remaining_capacity=self.simulator.receiver_buffer_capacity,
            read_thread=self.simulator.read_thread,
            network_thread=self.simulator.network_thread,
            write_thread=self.simulator.write_thread,
            history_length=self.history_length
        )
        
        self.current_step = 0
        self.trajectory = [self.state.copy()]

        # Return initial state as NumPy array
        return self.state.to_array()

class ResidualBlock(nn.Module):
    def __init__(self, size, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.activation = activation()

    def forward(self, x):
        # Save the input (for the skip connection)
        residual = x
        
        # Pass through two linear layers with activation
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        
        # Add the original input (residual connection)
        out += residual
        
        # Optionally add another activation at the end
        out = self.activation(out)
        return out
    
# class PolicyNetworkDiscrete(nn.Module):
#     def __init__(self, state_dim, num_actions=5):
#         super(PolicyNetworkDiscrete, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 256)
        
#         # Output layer. If you have 3 heads (for read/network/write),
#         # you produce 3*num_actions logits in total.
#         self.action_head = nn.Linear(256, 3 * num_actions)
        
#         self.to(device)
        
#     def forward(self, state):
#         """
#         state: shape [batch_size, state_dim]
#         Returns: logits of shape [batch_size, 3, num_actions]
#         """
#         x = torch.tanh(self.fc1(state))
#         x = torch.tanh(self.fc2(x))
#         x = torch.tanh(self.fc3(x))
#         x = self.action_head(x)
#         # Reshape to [batch_size, 3, num_actions]
#         return x.view(-1, 3, 5)
    
class PhysicsAwarePolicyDiscrete(nn.Module):
    def __init__(self, state_dim, num_actions=5):
        super().__init__()
        # Existing layers
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        
        # Action head remains
        self.action_head = nn.Linear(256, 3 * num_actions)
        
        # New components
        self.bottleneck_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Read/Network/Write bottleneck probs
            nn.Softmax(dim=-1)
        )
        
        self.grad_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # ∂reward/∂(read,net,write)
            nn.Tanh()  # Normalized gradients
        )
        
        self.to(device)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        # Original action logits
        action_logits = self.action_head(x).view(-1, 3, 5).float()
        
        # New outputs
        bottleneck_probs = self.bottleneck_head(x).float()
        gradients = self.grad_estimator(x).float()
        
        return action_logits, bottleneck_probs, gradients
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.value_head = nn.Linear(256, 1)
        self.to(device)

    def forward(self, state):
        """
        state: shape [batch_size, state_dim]
        Returns: value estimate of shape [batch_size, 1]
        """
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        value = self.value_head(x)
        return value

class PPOAgentDiscrete:
    def __init__(self, 
                 state_dim, 
                 lr=1e-3, 
                 gamma=0.99, 
                 eps_clip=0.2,
                 K_epochs=10,              # CHANGE #1: new arg
                 mini_batch_size=64):      # CHANGE #2: new arg
        self.policy = PhysicsAwarePolicyDiscrete(state_dim)
        self.policy_old = PhysicsAwarePolicyDiscrete(state_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_function = ValueNetwork(state_dim)
        # Add gradient normalizer
        self.grad_norm = nn.InstanceNorm1d(3)
        
        # Modified optimizer
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_function.parameters(), 'lr': lr},
            {'params': self.grad_norm.parameters(), 'lr': lr*0.1}
        ])
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()
        self.action_values = [-3, -1, 0, 1, 3]
        
        self.K_epochs = K_epochs                # store them
        self.mini_batch_size = mini_batch_size

    def select_action(self, state, is_inference=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)   # [1, obs_dim]
        logits, bottlenecks, gradients = self.policy_old(state)
            
        # Gradient-informed action modulation
        grad_impact = 0.3 * self.grad_norm(gradients.unsqueeze(0)).squeeze(0)
        modulated_logits = logits + grad_impact.unsqueeze(-1)
        probs = torch.softmax(modulated_logits, dim=-1)

        if is_inference:
            discrete_actions = torch.argmax(probs, dim=-1)  # [1, 3] each in {0..4}
        else:
            dist = torch.distributions.Categorical(probs)
            discrete_actions = dist.sample()                # [1, 3]

        # Convert discrete_actions -> log_probs
        log_probs = torch.log_softmax(logits, dim=-1)       # [1, 3, 5]
        chosen_log_probs = torch.gather(
            log_probs, dim=-1, index=discrete_actions.unsqueeze(-1)
        ).squeeze(-1)                                       # [1, 3]
        chosen_log_probs = chosen_log_probs.sum(dim=1)      # [1]
        logprob_scalar = chosen_log_probs.item()            # float
        bottleneck = torch.argmax(bottlenecks, dim=1).item() # int in {0,1,2}

        # Convert the discrete_actions => environment thread_changes
        # shape [1,3], so take row 0 => shape [3]
        actions_np = discrete_actions[0].cpu().numpy()         # e.g. [2,4,1]
        thread_changes = np.array(
            [self.action_values[a] for a in actions_np],    
            dtype=np.int32
        )  # e.g. if action_values=[-3,-1,0,1,3], then [0,3,1] => [0, +3, -1]

        return thread_changes, logprob_scalar, discrete_actions[0].cpu().numpy(), bottleneck, gradients.detach().cpu().numpy()[0]


    def update(self, memory):
        states = torch.stack(memory.states).to(device)   # shape [N, state_dim]
        actions = torch.tensor(memory.actions, dtype=torch.long).to(device)  # shape [N, 3]
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device) # shape [N]
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32).to(device) # shape [N]

        # ---- 1) Compute discounted returns ----
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Optionally normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Pre-compute state-values
        with torch.no_grad():
            state_values = self.value_function(states).squeeze()  # shape [N]

        # Advantage
        advantages = returns - state_values

        # ---- 2) Multiple epochs over the batch ----
        full_batch_size = len(states)
        indices = np.arange(full_batch_size)

        for _ in range(self.K_epochs):               # Repeat K_epochs
            np.random.shuffle(indices)

            for start in range(0, full_batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                # Extract mini-batch
                mb_states      = states[mb_indices]
                mb_actions     = actions[mb_indices]       # shape [MB, 3]
                mb_old_logprob = old_logprobs[mb_indices]  # shape [MB]
                mb_returns     = returns[mb_indices]       # shape [MB]
                mb_advantages  = advantages[mb_indices]    # shape [MB]

                # ---- Forward pass for new log-probs ----
                logits, bottlenecks, gradients = self.policy(mb_states)                    # shape [MB, 3, 5]
                new_logprobs_all = torch.log_softmax(logits, dim=-1)  # shape [MB, 3, 5]
                # Gather log-probs of chosen actions
                selected_logprobs = new_logprobs_all.gather(
                    2, mb_actions.unsqueeze(2)
                ).squeeze(-1)  # shape [MB, 3]

                # Sum across the 3 dimensions (read/network/write)
                new_logprobs = selected_logprobs.sum(dim=1)  # shape [MB]

                # Entropy
                probs_all = torch.softmax(logits, dim=-1)   # shape [MB, 3, 5]
                entropy_all = -(probs_all * new_logprobs_all).sum(dim=-1) # shape [MB, 3]
                entropy = entropy_all.sum(dim=1).mean()      # mean across mini-batch

                # Value
                V = self.value_function(mb_states).squeeze()  # shape [MB]

                # Surrogate ratio
                ratios = torch.exp(new_logprobs - mb_old_logprob)  # shape [MB]

                # PPO objectives
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                # Critic loss
                value_loss = self.MseLoss(V, mb_returns)

                # Actor loss: negative of clipped surrogate, plus value loss, minus entropy bonus
                actor_loss  = -torch.min(surr1, surr2).mean()

                subset_bottleneck_true = [memory.bottleneck_true[i] for i in mb_indices]
                subset_grad_true       = [memory.grad_true[i]       for i in mb_indices]

                bottleneck_loss = torch.nn.functional.cross_entropy(
                    bottlenecks,  # shape [MB, 3]
                    torch.tensor(subset_bottleneck_true).to(device)  # shape [MB]
                )

                grad_loss = torch.nn.functional.mse_loss(
                    gradients,     # shape [MB, 3]
                    torch.tensor(subset_grad_true).to(device)  # shape [MB, 3]
                )
                
                # Combined loss
                total_loss = (
                    actor_loss + 
                    0.5 * value_loss +
                    0.2 * bottleneck_loss +
                    0.3 * grad_loss -
                    0.01 * entropy
                )
                
                # Backprop through all components
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # ---- 3) After the multiple epochs, update old policy ----
        self.policy_old.load_state_dict(self.policy.state_dict())


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.bottleneck_true = []
        self.grad_true = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.bottleneck_true[:]
        del self.grad_true[:]


from tqdm import tqdm

def train_ppo(env, agent, max_episodes=1000, is_inference=False):
    memory = Memory()
    total_rewards = []
    for episode in tqdm(range(1, max_episodes + 1), desc="Episodes"):
        state = None
        simulator_generator = SimulatorGenerator()
        if episode % 500 == 0:
            _, simulator = simulator_generator.generate_simulator(episode=episode)
            state = env.reset(simulator=simulator)
        else:
            state = env.reset()
        episode_reward = 0
        for t in range(env.max_steps):
            thread_changes, logprob_scalar, action_indices, _, _ = agent.select_action(state)
        
            # Step environment with thread_changes
            next_state, reward, grads, bottleneck_idx, done, _ = env.step(thread_changes)

            memory.states.append(torch.FloatTensor(state).to(device))
            memory.actions.append(action_indices)       # This is crucial! action_indices in [0..4]
            memory.logprobs.append(logprob_scalar)
            memory.rewards.append(reward)
            memory.bottleneck_true.append(bottleneck_idx)
            memory.grad_true.append(grads)

            state = next_state
            if t==0:
                episode_reward += reward
            if done:
                break

        agent.update(memory)

        # print(f"Episode {episode}\tLast State: {state}\tReward: {reward}")
        with open('episode_rewards_training_dicrete_w_history_minibatch_mlp_deepseek_v7.csv', 'a') as f:
                f.write(f"Episode {episode}, Last State: {np.round(state[-3:])}, Reward: {reward}\n")

        memory.clear()
        total_rewards.append(episode_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
        if episode % 1000 == 0:
            save_model(agent, "models/training_dicrete_w_history_minibatch_mlp_deepseek_v7_policy_"+ str(episode) +".pth", "models/training_dicrete_w_history_minibatch_mlp_deepseek_v7_value_"+ str(episode) +".pth")
            print("Model saved successfully.")
    return total_rewards

def plot_rewards(rewards, title, pdf_file):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.xlim(0, len(rewards))
    plt.ylim(-1, 1)
    plt.title(title)
    plt.grid(True)
    
    plt.savefig(pdf_file)  
    plt.close()

import csv

import pandas as pd

def plot_threads_csv(threads_file='threads_dicrete_w_history_minibatch_mlp_deepseek_v7.csv', optimals = None, output_file='threads_plot.png'):
    optimal_read, optimal_network, optimal_write, _ = optimals
    data = []

    # Read data from threads_dicrete_w_history_minibatch_mlp_deepseek_v7.csv
    with open(threads_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            data.append([float(value) for value in row[:3]])

    df = pd.DataFrame(data, columns=['Read Threads', 'Network Threads', 'Write Threads'])

    # Compute rolling averages
    rolling_read = df['Read Threads'].rolling(window=15).mean()
    rolling_network = df['Network Threads'].rolling(window=15).mean()
    rolling_write = df['Write Threads'].rolling(window=15).mean()

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

    # save average thread count to a file
    with open('average_threads_dicrete_w_history_minibatch_mlp_deepseek_v7.csv', 'a') as f:
        f.write(f"optimal: {optimal_read}; Actual: {np.mean(df['Read Threads'])}\n")
        f.write(f"optimal: {optimal_network}; Actual: {np.mean(df['Network Threads'])}\n")
        f.write(f"optimal: {optimal_write}; Actual: {np.mean(df['Write Threads'])}\n")

# Function to plot throughputs with rolling averages
def plot_throughputs_csv(throughputs_file='throughputs_dicrete_w_history_minibatch_mlp_deepseek_v7.csv', optimals = None, output_file='throughputs_plot.png'):
    optimal_throughput = optimals[-1]
    data = []

    # Read data from throughputs_dicrete_w_history_minibatch_mlp_deepseek_v7.csv
    with open(throughputs_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            data.append([float(value) for value in row[:3]])

    df = pd.DataFrame(data, columns=['Read Throughput', 'Network Throughput', 'Write Throughput'])

    # Compute rolling averages
    rolling_read = df['Read Throughput'].rolling(window=15).mean()
    rolling_network = df['Network Throughput'].rolling(window=15).mean()
    rolling_write = df['Write Throughput'].rolling(window=15).mean()

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

    # save average throughput to a file
    with open('average_throughput_dicrete_w_history_minibatch_mlp_deepseek_v7.csv', 'a') as f:
        f.write(f"{np.mean(df['Read Throughput'])}\n")
        f.write(f"{np.mean(df['Network Throughput'])}\n")
        f.write(f"{np.mean(df['Write Throughput'])}\n")



import os
import re

def find_last_policy_model():
    models = os.listdir("models")
    models = [model for model in models if re.match(r'training_dicrete_w_history_minibatch_mlp_deepseek_v7_policy_\d+\.pth', model)]
    models.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return models[-1]

def find_last_value_model():
    models = os.listdir("models")
    models = [model for model in models if re.match(r'training_dicrete_w_history_minibatch_mlp_deepseek_v7_value_\d+\.pth', model)]
    models.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return models[-1]

if __name__ == '__main__':
    history_length = 3
    obs_dim = 5 + 7 * history_length

    if os.path.exists('threads_dicrete_w_history_minibatch_mlp_deepseek_v7.csv'):
        os.remove('threads_dicrete_w_history_minibatch_mlp_deepseek_v7.csv')
    if os.path.exists('throughputs_dicrete_w_history_minibatch_mlp_deepseek_v7.csv'):
        os.remove('throughputs_dicrete_w_history_minibatch_mlp_deepseek_v7.csv')

    oneGB = 1024
    simulator = NetworkSystemSimulator(sender_buffer_capacity=10*oneGB,
                                                receiver_buffer_capacity=6*oneGB,
                                                read_throughput_per_thread=200,
                                                network_throughput_per_thread=150,
                                                write_throughput_per_thread=70,
                                                read_bandwidth=12*oneGB,
                                                write_bandwidth=2*oneGB,
                                                network_bandwidth=2*oneGB,
                                                track_states=True)
    env = NetworkOptimizationEnv(simulator=simulator, history_length=history_length)
    agent = PPOAgentDiscrete(
        state_dim=obs_dim,
        lr=1e-4,
        eps_clip=0.1,
        K_epochs=10,         # e.g., 10 epochs
        mini_batch_size=32   # e.g., batch size of 32
    )
    rewards = train_ppo(env, agent, max_episodes=400000)
    
    plot_rewards(rewards, 'PPO Training Rewards', 'training_rewards_training_dicrete_w_history_minibatch_mlp_deepseek_v7.pdf')

    inference_count = 5
    simulator_generator = SimulatorGenerator()
    for i in range(inference_count):
        if os.path.exists('threads_dicrete_w_history_minibatch_mlp_deepseek_v7.csv'):
            os.remove('threads_dicrete_w_history_minibatch_mlp_deepseek_v7.csv')
        if os.path.exists('throughputs_dicrete_w_history_minibatch_mlp_deepseek_v7.csv'):
            os.remove('throughputs_dicrete_w_history_minibatch_mlp_deepseek_v7.csv')

        optimals, simulator = simulator_generator.generate_simulator(episode=500000)

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

        env = NetworkOptimizationEnv(simulator=simulator, history_length=history_length)
        agent = PPOAgentDiscrete(
            state_dim=obs_dim,
            lr=1e-4,
            eps_clip=0.1,
            K_epochs=10,         # e.g., 10 epochs
            mini_batch_size=32   # e.g., batch size of 32
        )

        policy_model = 'training_dicrete_w_history_minibatch_mlp_deepseek_v7_policy_400000.pth'
        value_model = 'training_dicrete_w_history_minibatch_mlp_deepseek_v7_value_400000.pth'

        print(f"Loading model... Value: {value_model}, Policy: {policy_model}")
        load_model(agent, "models/"+policy_model, "models/"+value_model)
        print("Model loaded successfully.")

        rewards = train_ppo(env, agent, max_episodes=100)

        plot_rewards(rewards, 'PPO Inference Rewards', 'rewards/inference_rewards_training_dicrete_w_history_minibatch_mlp_deepseek_v7_'+ str(i) +'.pdf')
        plot_threads_csv('threads_dicrete_w_history_minibatch_mlp_deepseek_v7.csv', optimals, 'threads/inference_threads_plot_training_dicrete_w_history_minibatch_mlp_deepseek_v7_'+ str(i) +'.png')
        plot_throughputs_csv('throughputs_dicrete_w_history_minibatch_mlp_deepseek_v7.csv', optimals, 'throughputs/inference_throughputs_plot_training_dicrete_w_history_minibatch_mlp_deepseek_v7_'+ str(i) +'.png') 