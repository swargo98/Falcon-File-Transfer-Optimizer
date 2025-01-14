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
import datetime

chunk_size = mp.Value("i", 1024*1024)
cpus = mp.Manager().list()

log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
log_file = "logs/" + datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"

if configurations["loglevel"] == "debug":
    log.basicConfig(
        format=log_FORMAT,
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=log.DEBUG,
        handlers=[
            log.FileHandler(log_file),
            log.StreamHandler()
        ]
    )

    mp.log_to_stderr(log.DEBUG)
else:
    log.basicConfig(
        format=log_FORMAT,
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=log.INFO
    )

    mp.log_to_stderr(log.INFO)


def worker(sock, process_num):
    while True:
        try:
            client, address = sock.accept()
            log.info("{u} connected".format(u=address))
            process_status[process_num] = 1
            total = 0
            d = client.recv(1).decode()
            while d:
                header = ""
                while d != '\n':
                    header += str(d)
                    d = client.recv(1).decode()

                if file_transfer:
                    file_stats = header.split(",")
                    filename, offset, to_rcv = str(file_stats[0]), int(file_stats[1]), int(file_stats[2])

                    if direct_io:
                        fd = os.open(root+filename, os.O_CREAT | os.O_RDWR | os.O_DIRECT | os.O_SYNC)
                        m = mmap.mmap(-1, to_rcv)
                    else:
                        fd = os.open(root+filename, os.O_CREAT | os.O_RDWR)

                    os.lseek(fd, offset, os.SEEK_SET)
                    log.debug("Receiving file: {0}".format(filename))
                    chunk = client.recv(chunk_size.value)

                    while chunk:
                        # log.debug("Chunk Size: {0}".format(len(chunk)))
                        if direct_io:
                            m.write(chunk)
                            os.write(fd, m)
                        else:
                            os.write(fd, chunk)

                        to_rcv -= len(chunk)
                        total += len(chunk)

                        if to_rcv > 0:
                            chunk = client.recv(min(chunk_size.value, to_rcv))
                        else:
                            log.debug("Successfully received file: {0}".format(filename))
                            break

                    os.close(fd)
                else:
                    chunk = client.recv(chunk_size.value)
                    while chunk:
                        chunk = client.recv(chunk_size.value)

                d = client.recv(1).decode()

            total = np.round(total/(1024*1024))
            log.info("{u} exited. total received {d} MB".format(u=address, d=total))
            client.close()
            process_status[process_num] = 0
        except Exception as e:
            log.error(str(e))
            # raise e


def report_throughput():
    global cpus
    time.sleep(1)

    while sum(process_status) > 0:
        t1 = time.time()
        cpus.append(psutil.cpu_percent())
        log.info(f"cpu: curr - {np.round(cpus[-1], 4)}, avg - {np.round(np.mean(cpus), 4)}")
        t2 = time.time()
        time.sleep(max(0, 1 - (t2-t1)))


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    parser=argparse.ArgumentParser()
    parser.add_argument("--host", help="Receiver Host Address")
    parser.add_argument("--port", help="Receiver Port Number")
    parser.add_argument("--data_dir", help="Receiver Data Directory")
    args = vars(parser.parse_args())
    # pp.pprint(f"Command line arguments: {args}")

    if args["host"]:
        configurations["receiver"]["host"] = args["host"]

    if args["port"]:
        configurations["receiver"]["port"] = int(args["port"])

    if args["data_dir"]:
        configurations["data_dir"] = args["data_dir"]

    configurations["cpu_count"] = mp.cpu_count()
    pp.pprint(configurations)

    direct_io = False
    root = configurations["data_dir"]
    HOST, PORT = configurations["receiver"]["host"], configurations["receiver"]["port"]

    file_transfer = True
    if "file_transfer" in configurations and configurations["file_transfer"] is not None:
        file_transfer = configurations["file_transfer"]

    num_workers = min(max(1,configurations["max_cc"]), configurations["cpu_count"])

    sock = socket.socket()
    sock.bind((HOST, PORT))
    sock.listen(num_workers)

    iter = 0
    while iter<1:
        iter += 1
        log.info(">>>>>> Iterations: {0} >>>>>>".format(iter))

        process_status = mp.Array("i", [0 for _ in range(num_workers)])
        workers = [mp.Process(target=worker, args=(sock, i,)) for i in range(num_workers)]
        for p in workers:
            p.daemon = True
            p.start()

        # while True:
        #     try:
        #         time.sleep(1)
        #     except:
        #         break

        process_status[0] = 1
        # reporting_process = mp.Process(target=report_throughput)
        # reporting_process.daemon = True
        # reporting_process.start()

        while sum(process_status) > 0:
            time.sleep(0.1)

        # reporting_process.terminate()
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.1)
