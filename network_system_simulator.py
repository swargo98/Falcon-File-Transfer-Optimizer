from scipy.optimize import minimize, differential_evolution
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
        read_thread, network_thread, write_thread = map(int, threads)
        self.read_thread = read_thread
        self.network_thread = network_thread
        self.write_thread = write_thread

        # sender buffer simulation
        read_and_network_thread = np.lcm(int(read_thread), int(network_thread))
        read_throughput = 0
        network_throughput_sender = 0
        sender_buffer_before = self.sender_buffer_in_use

        for i in range(read_and_network_thread):
            if i % read_thread == 0 and self.sender_buffer_in_use < self.sender_buffer_capacity:
                read_throughput_temp = min(self.read_throughput_per_thread, self.sender_buffer_capacity - self.sender_buffer_in_use)
                read_throughput += min(read_throughput_temp, self.read_bandwidth-read_throughput)
                self.sender_buffer_in_use += min(read_throughput_temp, self.read_bandwidth-read_throughput)

            if i % network_thread == 0 and self.sender_buffer_in_use > 0:
                network_throughput_temp = min(self.network_throughput_per_thread, self.sender_buffer_in_use)
                network_throughput_sender += min(network_throughput_temp, self.network_bandwidth - network_throughput_sender)
                self.sender_buffer_in_use -= min(network_throughput_temp, self.network_bandwidth - network_throughput_sender)

        # receiver buffer simulation
        network_and_write_thread = np.lcm(int(network_thread), int(write_thread))
        write_throughput = 0
        network_throughput_receiver = 0
        receiver_buffer_before = self.receiver_buffer_in_use

        for i in range(network_and_write_thread):
            if i % network_thread == 0 and self.receiver_buffer_in_use < self.receiver_buffer_capacity:
                network_throughput_temp = min(self.network_throughput_per_thread, self.receiver_buffer_capacity - self.receiver_buffer_in_use)
                network_throughput_receiver += min(network_throughput_temp, self.network_bandwidth - network_throughput_receiver)
                self.receiver_buffer_in_use += min(network_throughput_temp, self.network_bandwidth - network_throughput_receiver)

            if i % write_thread == 0 and self.receiver_buffer_in_use > 0:
                write_throughput_temp = min(self.write_throughput_per_thread, self.receiver_buffer_in_use)
                write_throughput += min(write_throughput_temp, self.write_bandwidth - write_throughput)
                self.receiver_buffer_in_use -= min(write_throughput_temp, self.write_bandwidth - write_throughput)

        network_throughput = min(network_throughput_sender, network_throughput_receiver)

        if network_throughput == network_throughput_sender:
            write_throughput = min(write_throughput, network_throughput + receiver_buffer_before)
            self.receiver_buffer_in_use = max(receiver_buffer_before - write_throughput, 0)
        else:
            read_throughput = min(read_throughput, network_throughput + sender_buffer_before)
            self.sender_buffer_in_use = max(sender_buffer_before - read_throughput, 0)

        self.sender_buffer_in_use = max(self.sender_buffer_in_use, 0)
        self.receiver_buffer_in_use = max(self.receiver_buffer_in_use, 0)

        utility = (read_throughput/self.K ** read_thread) + (network_throughput/self.K ** network_thread) + (write_throughput/self.K ** write_thread)

        print(f"Read thread: {read_thread}, Network thread: {network_thread}, Write thread: {write_thread}, Utility: {utility}, Read throughput: {read_throughput}, Network throughput: {network_throughput}, Write throughput: {write_throughput}")

        return utility * -1
    
    def optimize_threads(self, init_threads=(1, 1, 1)):
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

# Example usage:
simulator = NetworkSystemSimulator()
optimal_threads = simulator.optimize_threads()
print(f"Optimal threads: Read={(optimal_threads[0])}, Network={(optimal_threads[1])}, Write={(optimal_threads[2])}")