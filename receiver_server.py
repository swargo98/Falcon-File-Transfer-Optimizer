import os
import mmap
import time
import socket
import pprint
import argparse
import logging as log
import numpy as np
import psutil
import multiprocessing as mp
from config_receiver import configurations

class ReceiverServer:
    def __init__(self, configurations):
        self.configurations = configurations

        # Set up logging
        log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
        if configurations["loglevel"] == "debug":
            log.basicConfig(
                format=log_FORMAT,
                datefmt='%m/%d/%Y %I:%M:%S %p',
                level=log.DEBUG,
            )
            mp.log_to_stderr(log.DEBUG)
        else:
            log.basicConfig(
                format=log_FORMAT,
                datefmt='%m/%d/%Y %I:%M:%S %p',
                level=log.INFO
            )
            mp.log_to_stderr(log.INFO)
        
        self.chunk_size = mp.Value("i", 1024*1024)
        self.cpus = mp.Manager().list()
        
        # Initialize other variables
        self.direct_io = False
        self.root = configurations["data_dir"]
        self.HOST = configurations["receiver"]["host"]
        self.PORT = configurations["receiver"]["port"]
        self.file_transfer = True
        if "file_transfer" in configurations and configurations["file_transfer"] is not None:
            self.file_transfer = configurations["file_transfer"]
        
        self.num_workers = min(max(1, configurations["max_cc"]), configurations["cpu_count"])
        self.sock = socket.socket()
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen(self.num_workers)
        self.process_status = mp.Array("i", [0 for _ in range(self.num_workers)])
        
    def start(self):
        iter = 0
        while iter < 1:
            iter +=1
            log.info(">>>>>> Iterations: {0} >>>>>>".format(iter))
            
            workers = [mp.Process(target=self.worker, args=(self.sock, i,)) for i in range(self.num_workers)]
            for p in workers:
                p.daemon = True
                p.start()
            
            self.process_status[0] = 1
            while sum(self.process_status) > 0:
                time.sleep(0.1)
            
            for p in workers:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=0.1)
                    
    def worker(self, sock, process_num):
        while True:
            try:
                client, address = sock.accept()
                log.info("{u} connected".format(u=address))
                self.process_status[process_num] = 1
                total = 0
                d = client.recv(1).decode()
                while d:
                    header = ""
                    while d != '\n':
                        header += str(d)
                        d = client.recv(1).decode()
                    
                    if self.file_transfer:
                        file_stats = header.split(",")
                        filename, offset, to_rcv = str(file_stats[0]), int(file_stats[1]), int(file_stats[2])
                        if self.direct_io:
                            fd = os.open(self.root+filename, os.O_CREAT | os.O_RDWR | os.O_DIRECT | os.O_SYNC)
                            m = mmap.mmap(-1, to_rcv)
                        else:
                            fd = os.open(self.root+filename, os.O_CREAT | os.O_RDWR)
                        os.lseek(fd, offset, os.SEEK_SET)
                        log.debug("Receiving file: {0}".format(filename))
                        chunk = client.recv(self.chunk_size.value)
                        while chunk:
                            if self.direct_io:
                                m.write(chunk)
                                os.write(fd, m)
                            else:
                                os.write(fd, chunk)
                            to_rcv -= len(chunk)
                            total += len(chunk)
                            if to_rcv > 0:
                                chunk = client.recv(min(self.chunk_size.value, to_rcv))
                            else:
                                log.debug("Successfully received file: {0}".format(filename))
                                break
                        os.close(fd)
                    else:
                        chunk = client.recv(self.chunk_size.value)
                        while chunk:
                            chunk = client.recv(self.chunk_size.value)
                    d = client.recv(1).decode()
                total = np.round(total/(1024*1024))
                log.info("{u} exited. total received {d} MB".format(u=address, d=total))
                client.close()
                self.process_status[process_num] = 0
            except Exception as e:
                log.error(str(e))
                
    def report_throughput(self):
        time.sleep(1)
        while sum(self.process_status) > 0:
            t1 = time.time()
            self.cpus.append(psutil.cpu_percent())
            log.info(f"cpu: curr - {np.round(self.cpus[-1], 4)}, avg - {np.round(np.mean(self.cpus), 4)}")
            t2 = time.time()
            time.sleep(max(0, 1 - (t2 - t1)))
    
if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Receiver Host Address")
    parser.add_argument("--port", help="Receiver Port Number")
    parser.add_argument("--data_dir", help="Receiver Data Directory")
    args = vars(parser.parse_args())

    if args["host"]:
        configurations["receiver"]["host"] = args["host"]
    if args["port"]:
        configurations["receiver"]["port"] = int(args["port"])
    if args["data_dir"]:
        configurations["data_dir"] = args["data_dir"]

    configurations["cpu_count"] = mp.cpu_count()
    pp.pprint(configurations)
    
    server = ReceiverServer(configurations)
    server.start()