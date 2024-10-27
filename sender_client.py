import os
import time
import uuid
import socket
import warnings
import datetime
import numpy as np
import psutil
import pprint
import argparse
import logging as log
import multiprocessing as mp
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from config_sender import configurations
from optimizers import BayesOptimizer, BruteForceOptimizer, GradientDescentOptimizer, HillClimbingOptimizer, CGOptimizer, BinarySearchOptimizer

class SenderClient:
    def __init__(self, configurations):
        self.configurations = configurations
        self.setup_logging()

        # Initialize variables
        self.emulab_test = configurations.get("emulab_test", False)
        self.centralized = configurations.get("centralized", False)
        self.file_transfer = configurations.get("file_transfer", True)
        self.chunk_size = 1 * 1024 * 1024  # 1 MB
        self.exit_signal = 10 ** 10
        self.probing_time = configurations["probing_sec"]
        self.HOST = configurations["receiver"]["host"]
        self.PORT = configurations["receiver"]["port"]
        self.RECEIVER_ADDRESS = f"{self.HOST}:{self.PORT}"
        self.root = configurations["data_dir"]
        self.throughput_logs = mp.Manager().list()
        self.cpus = mp.Manager().list()
        self.num_workers = mp.Value("i", 0)
        self.file_incomplete = mp.Value("i", 0)
        self.process_status = mp.Array("i", [0 for _ in range(configurations["thread_limit"])])
        self.file_offsets = None  # To be initialized after loading files
        self.q = None # To be initialized after loading files
        self.workers = []
        self.reporting_process = None

        # Load files
        self.load_files()

    def setup_logging(self):
        """Set up logging configuration."""
        log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
        log_file = "logs/" + datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
        if self.configurations["loglevel"] == "debug":
            log_level = log.DEBUG
        else:
            log_level = log.INFO

        log.basicConfig(
            format=log_FORMAT,
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=log_level,
            handlers=[
                log.FileHandler(log_file),
                log.StreamHandler()
            ]
        )
        mp.log_to_stderr(log_level)

    def load_files(self):
        """Load file names and sizes."""
        self.file_names = os.listdir(self.root) * self.configurations["multiplier"]
        self.file_sizes = [os.path.getsize(os.path.join(self.root, filename)) for filename in self.file_names]
        self.file_count = len(self.file_names)
        self.file_incomplete.value = self.file_count
        self.file_offsets = mp.Array("d", [0.0 for _ in range(self.file_count)])
        
        self.q = mp.Manager().Queue(maxsize=self.file_count)
        for i in range(self.file_count):
            self.q.put(i)

    def tcp_stats(self):
        """Collect TCP statistics."""
        start = time.time()
        sent, retransmissions = 0, 0
        try:
            data = os.popen("ss -ti").read().split("\n")
            for i in range(1, len(data)):
                if self.RECEIVER_ADDRESS in data[i-1]:
                    parse_data = data[i].split(" ")
                    for entry in parse_data:
                        if "data_segs_out" in entry:
                            sent += int(entry.split(":")[-1])
                        elif "retrans" in entry:
                            retransmissions += int(entry.split("/")[-1])
        except Exception as e:
            log.error(f"Error collecting TCP stats: {e}")
        end = time.time()
        log.debug("Time taken to collect tcp stats: {0}ms".format(np.round((end - start) * 1000)))
        return sent, retransmissions

    def worker(self, process_id):
        """Worker process to handle file sending."""
        self.setup_logging()
        while self.file_incomplete.value > 0:
            if self.process_status[process_id] == 0:
                pass
            else:
                while self.num_workers.value < 1:
                    pass

                log.debug(f"Start Process :: {process_id}")
                try:
                    sock = socket.socket()
                    sock.settimeout(3)
                    sock.connect((self.HOST, self.PORT))

                    if self.emulab_test:
                        target, factor = 2500, 10
                        max_speed = (target * 1000 * 1000) / 8
                        second_target, second_data_count = int(max_speed / factor), 0

                    while (not self.q.empty()) and (self.process_status[process_id] == 1):
                        try:
                            file_id = self.q.get()
                        except:
                            self.process_status[process_id] = 0
                            break

                        offset = self.file_offsets[file_id]
                        to_send = self.file_sizes[file_id] - offset

                        if (to_send > 0) and (self.process_status[process_id] == 1):
                            filename = os.path.join(self.root, self.file_names[file_id])
                            with open(filename, "rb") as file:
                                msg = f"{self.file_names[file_id]},{int(offset)},{int(to_send)}\n"
                                sock.send(msg.encode())
                                log.debug(f"Starting {process_id}, {file_id}, {filename}")
                                timer100ms = time.time()

                                while (to_send > 0) and (self.process_status[process_id] == 1):
                                    if self.emulab_test:
                                        block_size = min(self.chunk_size, second_target - second_data_count)
                                        data_to_send = bytearray(int(block_size))
                                        sent = sock.send(data_to_send)
                                    else:
                                        block_size = min(self.chunk_size, to_send)
                                        if self.file_transfer:
                                            sent = sock.sendfile(file=file, offset=int(offset), count=int(block_size))
                                        else:
                                            data_to_send = bytearray(int(block_size))
                                            sent = sock.send(data_to_send)

                                    offset += sent
                                    to_send -= sent
                                    self.file_offsets[file_id] = offset

                                    if self.emulab_test:
                                        second_data_count += sent
                                        if second_data_count >= second_target:
                                            second_data_count = 0
                                            while timer100ms + (1 / factor) > time.time():
                                                pass
                                            timer100ms = time.time()

                        if to_send > 0:
                            self.q.put(file_id)
                        else:
                            with self.file_incomplete.get_lock():
                                self.file_incomplete.value -= 1

                    sock.close()

                except socket.timeout:
                    pass

                except Exception as e:
                    self.process_status[process_id] = 0
                    log.debug(f"Process: {process_id}, Error: {str(e)}")

                log.debug(f"End Process :: {process_id}")

        self.process_status[process_id] = 0

    def sample_transfer(self, parameters):
        """Sample transfer for optimization."""
        if self.file_incomplete.value == 0:
            return self.exit_signal

        # Ensure parameters are at least 1 and rounded to integers
        parameters = [max(1, int(np.round(x))) for x in parameters]
        log.info(f"Sample Transfer -- Probing Parameters: {parameters}")
        self.num_workers.value = parameters[0]

        current_concurrency = np.sum(self.process_status)
        for i in range(self.configurations["thread_limit"]):
            if i < parameters[0]:
                if i >= current_concurrency:
                    self.process_status[i] = 1
            else:
                self.process_status[i] = 0

        log.debug(f"Active Concurrency Level: {np.sum(self.process_status)}")

        time.sleep(1)
        prev_sent_count, prev_retrans_count = self.tcp_stats()
        probe_end_time = time.time() + self.probing_time - 1.1

        while (time.time() < probe_end_time) and (self.file_incomplete.value > 0):
            time.sleep(0.1)

        curr_sent_count, curr_retrans_count = self.tcp_stats()
        sent_count_diff = curr_sent_count - prev_sent_count
        retrans_count_diff = curr_retrans_count - prev_retrans_count

        log.debug(f"TCP Segments >> Sent Count: {sent_count_diff}, Retransmission Count: {retrans_count_diff}")
        throughput = np.mean(self.throughput_logs[-2:]) if len(self.throughput_logs) > 2 else 0

        loss_rate = 0
        B = int(self.configurations["B"])
        K = float(self.configurations["K"])
        if sent_count_diff != 0:
            loss_rate = retrans_count_diff / sent_count_diff if sent_count_diff > retrans_count_diff else 0

        loss_rate_impact = B * loss_rate
        concurrency_impact_nonlinear = K ** self.num_workers.value
        score = (throughput / concurrency_impact_nonlinear) - (throughput * loss_rate_impact)
        adjusted_score = np.round(score * (-1))

        log.info(
            f"Sample Transfer -- Throughput: {np.round(throughput)}Mbps, "
            f"Loss Rate: {np.round(loss_rate * 100, 2)}%, Score: {adjusted_score}"
        )

        if self.file_incomplete.value == 0:
            return self.exit_signal
        else:
            return adjusted_score

    def normal_transfer(self, parameteters):
        """Run normal transfer after optimization."""
        self.num_workers.value = max(1, int(np.round(parameteters)[0]))
        log.info(f"Normal Transfer -- Probing Parameters: {[self.num_workers.value]}")

        for i in range(self.num_workers.value):
            self.process_status[i] = 1

        while (np.sum(self.process_status) > 0) and (self.file_incomplete.value > 0):
            pass

    def run_transfer(self):
        """Run the transfer process."""
        params = [2]

        if self.configurations["method"].lower() == "brute":
            log.info("Running Brute Force Optimization .... ")
            params = BruteForceOptimizer(self.configurations, self.sample_transfer, log).optimize()

        elif self.configurations["method"].lower() == "hill_climb":
            log.info("Running Hill Climb Optimization .... ")
            params = HillClimbingOptimizer(self.configurations, self.sample_transfer, log).optimize()

        elif self.configurations["method"].lower() == "gradient":
            log.info("Running Gradient Optimization .... ")
            params = GradientDescentOptimizer(self.configurations, self.sample_transfer, log).optimize()

        elif self.configurations["method"].lower() == "binary":
            log.info("Running Binary Search Optimization .... ")
            params = BinarySearchOptimizer(self.configurations, self.sample_transfer, log).optimize()

        elif self.configurations["method"].lower() == "cg":
            log.info("Running Conjugate Optimization .... ")
            params = CGOptimizer(self.configurations, self.sample_transfer, log).optimize()

        elif self.configurations["method"].lower() == "probe":
            log.info("Running a fixed configurations Probing .... ")
            params = [self.configurations["fixed_probing"]["thread"]]

        else:
            log.info("Running Bayesian Optimization .... ")
            params = BayesOptimizer(self.configurations, self.sample_transfer, log).optimize()

        if self.file_incomplete.value > 0:
            self.normal_transfer(params)

    def report_throughput(self, start_time):
        """Report throughput periodically."""
        self.setup_logging()
        previous_total = 0
        previous_time = 0

        while self.file_incomplete.value > 0:
            t1 = time.time()
            time_since_beginning = np.round(t1 - start_time, 1)

            if time_since_beginning >= 0.1:
                if time_since_beginning >= 3 and sum(self.throughput_logs[-3:]) == 0:
                    self.file_incomplete.value = 0

                total_bytes = np.sum(self.file_offsets)
                thrpt = np.round((total_bytes * 8) / (time_since_beginning * 1000 * 1000), 2)

                curr_total = total_bytes - previous_total
                curr_time_sec = np.round(time_since_beginning - previous_time, 3)
                curr_thrpt = np.round((curr_total * 8) / (curr_time_sec * 1000 * 1000), 2)
                previous_time, previous_total = time_since_beginning, total_bytes
                self.throughput_logs.append(curr_thrpt)
                m_avg = np.round(np.mean(self.throughput_logs[-60:]), 2)

                log.info(f"Throughput @{time_since_beginning}s: Current: {curr_thrpt}Mbps, Average: {thrpt}Mbps, 60Sec_Average: {m_avg}Mbps")

                t2 = time.time()
                time.sleep(max(0, 1 - (t2 - t1)))

    def start(self):
        """Start the sender client."""
        # Start worker processes
        for i in range(self.configurations["thread_limit"]):
            p = mp.Process(target=self.worker, args=(i,))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Start throughput reporting process
        start_time = time.time()
        self.reporting_process = mp.Process(target=self.report_throughput, args=(start_time,))
        self.reporting_process.daemon = True
        self.reporting_process.start()

        # Run the transfer
        self.run_transfer()

        # Wait for workers to finish
        for p in self.workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.1)

        # Terminate reporting process
        if self.reporting_process.is_alive():
            self.reporting_process.terminate()

        # Final throughput report
        end_time = time.time()
        time_since_beginning = np.round(end_time - start_time, 3)
        total = np.round(np.sum(self.file_offsets) / (1024 * 1024 * 1024), 3)
        thrpt = np.round((total * 8 * 1024) / time_since_beginning, 2)
        log.info(f"Total: {total} GB, Time: {time_since_beginning} sec, Throughput: {thrpt} Mbps")

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Receiver Host Address")
    parser.add_argument("--port", help="Receiver Port Number")
    parser.add_argument("--data_dir", help="Sender Data Directory")
    parser.add_argument("--method", help="choose one of them : gradient, bayes, brute, probe")
    args = vars(parser.parse_args())

    # Update configurations with command-line arguments
    if args["host"]:
        configurations["receiver"]["host"] = args["host"]
    if args["port"]:
        configurations["receiver"]["port"] = int(args["port"])
    if args["data_dir"]:
        configurations["data_dir"] = args["data_dir"]
    if args["method"]:
        configurations["method"] = args["method"]

    configurations["cpu_count"] = mp.cpu_count()
    configurations["thread_limit"] = min(max(1, configurations["max_cc"]), configurations["cpu_count"])
    pp.pprint(configurations)

    sender = SenderClient(configurations)
    sender.start()