from scipy.optimize import minimize, differential_evolution   
import numpy as np
from queue import PriorityQueue

from skopt.space import Integer
from skopt import Optimizer
import time

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
        next_time = time + time_taken + 0.00001
        if next_time < 1:
            self.thread_queue.put((next_time, "read"))
        
        if throughput_increase > 0 and self.track_states:
            with open('thread_level_states.csv', 'a') as f:
                f.write(f"Read, {throughput_increase}, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")
        return next_time

    def network_thread_task(self, time):
        throughput_increase = 0
        if self.sender_buffer_in_use > 0 and self.receiver_buffer_in_use < self.receiver_buffer_capacity:
            network_throughput_temp = min(self.network_throughput_per_thread, self.sender_buffer_in_use, self.receiver_buffer_capacity - self.receiver_buffer_in_use)
            throughput_increase = min(network_throughput_temp, self.network_bandwidth-self.network_throughput)
            self.network_throughput += throughput_increase
            self.sender_buffer_in_use -= throughput_increase
            self.receiver_buffer_in_use += throughput_increase

        time_taken = throughput_increase / self.network_throughput_per_thread
        next_time = time + time_taken + 0.00001
        if next_time < 1:
            self.thread_queue.put((next_time, "network"))
        
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
        next_time = time + time_taken + 0.00001
        if next_time < 1:
            self.thread_queue.put((next_time, "write"))
        
        if throughput_increase > 0 and self.track_states:
            with open('thread_level_states.csv', 'a') as f:
                f.write(f"Write, {throughput_increase}, {self.sender_buffer_in_use}, {self.receiver_buffer_in_use}\n")
        return next_time
    
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

        print(f"Read thread: {read_thread}, Network thread: {network_thread}, Write thread: {write_thread}, Utility: {utility}")
        
        if self.track_states:
            with open('optimizer_call_level_states.csv', 'a') as f:
                f.write(f"{read_thread}, {network_thread}, {write_thread}, {utility}, {self.read_throughput}, {self.sender_buffer_in_use}, {self.network_throughput}, {self.receiver_buffer_in_use}, {self.write_throughput}\n")
        
        return utility * -1
    
    def optimize_evolution(self, init_threads=(1, 1, 1)):
        bounds = [(1, 10), (1, 10), (1, 10)]  # Adjust the upper bound as needed

        result = differential_evolution(
            self.get_utility_value,
            bounds,
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=True
        )
        
        optimal_threads = np.round(result.x).astype(int)  # Ensure integer threads
        return optimal_threads
    
    def optimize_bayes(self):
        observation_limit, count = 25, 0
        search_space  = [
                Integer(1, 10),
                Integer(1, 10),
                Integer(5, 20),
            ]

        params = []
        optimizer = Optimizer(
            dimensions=search_space,
            base_estimator="GP",  # Gaussian Process (GP)
            acq_func="gp_hedge",  # Acquisition function (gp_hedge)
            acq_optimizer="auto",  # Acquisition optimizer (auto)
            n_random_starts=3,
            model_queue_size=observation_limit,
        )

        current_utility = 0
        previous_utility = 0
        patience = 10

        best_params = None
        best_score = None
        best_iteration = 0

        while True:
            previous_utility = current_utility

            if len(optimizer.yi) > observation_limit:
                optimizer.yi = optimizer.yi[-observation_limit:]
                optimizer.Xi = optimizer.Xi[-observation_limit:]

            # if self.verbose:
            #     self.logger.info(f"Iteration {count} Starts...")

            t1 = time.time()
            res = optimizer.run(func=self.get_utility_value, n_iter=3)
            t2 = time.time()

            # if self.verbose:
            print(f"Iteration {count} Ends, Best Params: {res.x} and Score: {res.fun * -1}.")

            # write the previous print statement to a file
            with open('optimization_log.csv', 'a') as f:
                f.write(f"Iteration {count} Ends, Best Params: {res.x} and Score: {res.fun * -1}.\n")

            last_utility_value = np.min(optimizer.yi[-3:])
            
            current_utility = last_utility_value

            # stop if convergence is reached
            if current_utility > previous_utility:
                print(f"Current Utility: {current_utility} is greater than Previous Utility: {previous_utility}.")
                count += 1

            if best_score is None or best_score > last_utility_value:
                best_score = -last_utility_value
                best_params = res.x
                best_iteration = count

            if patience == count:
                print(f"Stopping the optimization as the utility value is not improving for {patience} iterations.")
                print(f"Best Params: {best_params} and Best Score: {best_score} at iteration {best_iteration}.")
                params = res.x
                break

        return params
    

#############################################################

# import numpy as np
# import random

# class GradientDescentOptimizer:
#     def __init__(self, simulator, max_thread_counts=(10, 10, 10), exploration_rate=0.2, exploration_decay=0.95):
#         self.simulator = simulator
#         self.max_thread_counts = max_thread_counts  # Maximum thread counts for each type
#         self.exploration_rate = exploration_rate  # Initial exploration rate
#         self.exploration_decay = exploration_decay  # Exploration decay over iterations

#     def optimize(self):
#         max_thread_counts_local = 10
#         counts = [0, 0, 0]
#         min_cost = float('inf')
#         utility_values = []
#         concurrencies = [[1, 1, 1], [1, 1, 1]]  # Initial concurrency configuration
#         directions = [0, 0, 0]  # Separate direction for each variable

#         while True:
#             counts[0] += 1
#             current_concurrency = concurrencies[-1]
#             utility = self.simulator.get_utility_value(current_concurrency)
#             utility_values.append(utility)

#             if utility < min_cost:
#                 min_cost = utility  # Track best utility
#             else:
#                 # Apply exploration randomly to avoid local minima
#                 if random.random() < self.exploration_rate:
#                     # Introduce random perturbations within bounds
#                     current_concurrency = [
#                         min(max(1, current_concurrency[i] + random.randint(-2, 2)), max_thread_counts_local) for i in range(3)
#                     ]
#                     print(f"Exploration step at iteration {counts[0]}, new concurrency: {current_concurrency}")

#             # Compute gradient and update directions
#             if len(utility_values) > 1:
#                 for i in range(3):  # For read, network, write threads
#                     prev_concurrency = concurrencies[-1][i]
#                     current_concurrency_val = current_concurrency[i]

#                     # Calculate gradient
#                     gradient = (utility_values[-1] - utility_values[-2]) / (1 if prev_concurrency == current_concurrency_val else current_concurrency_val - prev_concurrency)
#                     gradient_change = np.abs(gradient / (1 if utility_values[-2] == 0 else utility_values[-2]))

#                     # Adjust direction based on gradient sign
#                     if gradient > 0:
#                         directions[i] = min(-1, directions[i] - 1)
#                     else:
#                         directions[i] = max(1, directions[i] + 1)

#                     # Calculate next step
#                     update = int(directions[i] * np.ceil(current_concurrency[i] * gradient_change))
#                     current_concurrency[i] = max(1, min(current_concurrency[i] + update, max_thread_counts_local))

#                 # Append the fully updated concurrency configuration
#                 concurrencies.append(current_concurrency.copy())

#             # Decay exploration rate gradually
#             # self.exploration_rate *= self.exploration_decay

#             # Stop criteria
#             if min(utility_values) < -16.5:  # Increased to 50 iterations for thorough exploration
#                 print(f"Stopping optimization. Best configuration found: {concurrencies[-1]} with utility: {min_cost}.")
#                 break

#         return concurrencies[-1]

# # Example usage:
# simulator = NetworkSystemSimulator()
# optimizer = GradientDescentOptimizer(simulator)
# optimal_threads = optimizer.optimize()
# print(f"Optimal threads: Read={(optimal_threads[0])}, Network={(optimal_threads[1])}, Write={(optimal_threads[2])}")

#############################################################
# Example usage:
simulator = NetworkSystemSimulator()
# optimal_threads = simulator.optimize_evolution()
optimal_threads = simulator.optimize_bayes()
print(f"Optimal threads: Read={(optimal_threads[0])}, Network={(optimal_threads[1])}, Write={(optimal_threads[2])}")
# simulator.get_utility_value([2, 3, 2])