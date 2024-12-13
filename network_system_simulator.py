from scipy.optimize import minimize
import numpy as np

class NetworkSystemSimulator:
    def __init__(self, read_thread = 1, network_thread = 1, write_thread = 1, sender_buffer_capacity = 10, receiver_buffer_capacity = 10, read_throughput_per_thread = 3, write_throughput_per_thread = 3, network_throughput_per_thread = 1, read_bandwidth = 6, write_bandwidth = 6, network_bandwidth = 6, read_background_traffic = 0, write_background_traffic = 0, network_background_traffic = 0):
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
        self.K = 1.02
        
        # Initialize the buffers
        self.sender_buffer_in_use = max(min(self.read_throughput_per_thread * read_thread - self.network_throughput_per_thread * self.network_thread, self.sender_buffer_capacity), 0)
        self.receiver_buffer_in_use = max(min(self.network_throughput_per_thread * network_thread - self.write_throughput_per_thread * self.write_thread, self.receiver_buffer_capacity), 0)

    def get_utility_value(self, threads):
        read_thread, network_thread, write_thread = threads
        self.read_thread = read_thread
        self.network_thread = network_thread
        self.write_thread = write_thread

        read_throughput = min(self.read_throughput_per_thread * read_thread, self.read_bandwidth)
        network_throughput = min(self.network_throughput_per_thread * network_thread, self.network_bandwidth)
        write_throughput = min(self.write_throughput_per_thread * write_thread, self.write_bandwidth)

        # Flow chart shared by Dr. Md Arifuzzaman (should it be here?)

        if read_throughput < network_throughput:
            network_throughput = read_throughput
        elif self.sender_buffer_in_use == self.sender_buffer_capacity:
            read_throughput = network_throughput

        if write_throughput > network_throughput:
            write_throughput = network_throughput
        elif self.receiver_buffer_in_use == self.receiver_buffer_capacity:
            network_throughput = write_throughput

        self.sender_buffer_in_use += read_throughput - network_throughput
        self.receiver_buffer_in_use += network_throughput - write_throughput

        self.sender_buffer_in_use = max(self.sender_buffer_in_use, 0)
        self.receiver_buffer_in_use = max(self.receiver_buffer_in_use, 0)

        utility = (read_throughput/self.K ** read_thread) + (network_throughput/self.K ** network_thread) + (write_throughput/self.K ** write_thread)

        print(f"Read thread: {read_thread}, Network thread: {network_thread}, Write thread: {write_thread}, Utility: {utility}, Read throughput: {read_throughput}, Network throughput: {network_throughput}, Write throughput: {write_throughput}")

        return utility * -1
    
    def optimize_threads(self, init_threads=(1, 1, 1)):
        result = minimize(self.get_utility_value, init_threads, method='CG')
        optimal_threads = result.x
        return optimal_threads

# Example usage:
simulator = NetworkSystemSimulator()
optimal_threads = simulator.optimize_threads()
print(f"Optimal threads: Read={(optimal_threads[0])}, Network={(optimal_threads[1])}, Write={(optimal_threads[2])}")