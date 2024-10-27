import socket
import argparse
import os

def send_file(receiver_host, receiver_port, source_file, dest_filename, offset=0):
    """
    Connects to the receiver server and sends a file.

    :param receiver_host: Hostname or IP address of the receiver server
    :param receiver_port: Port number of the receiver server
    :param source_file: Path to the source file to send
    :param dest_filename: Destination filename on the receiver side
    :param offset: Offset in bytes to start sending from (default is 0)
    """
    try:
        # Create a socket connection to the receiver
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((receiver_host, receiver_port))
        print(f"Connected to receiver at {receiver_host}:{receiver_port}")

        # Get the size of the file to send
        file_size = os.path.getsize(source_file) - offset
        if file_size <= 0:
            print("File size is zero or negative after applying offset.")
            return

        # Prepare and send the header
        header = f"{dest_filename},{offset},{file_size}\n"
        print(f"Sending header: {header.strip()}")
        sock.sendall(header.encode())

        # Send the file data in chunks
        with open(source_file, 'rb') as f:
            f.seek(offset)
            bytes_sent = 0
            while bytes_sent < file_size:
                chunk_size = min(1024 * 1024, file_size - bytes_sent)  # 1 MB chunks
                data = f.read(chunk_size)
                if not data:
                    break
                sock.sendall(data)
                bytes_sent += len(data)
                print(f"Sent {bytes_sent}/{file_size} bytes", end='\r')

        print("\nFile transfer completed.")
        sock.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        if sock:
            sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a file to the receiver server.")
    parser.add_argument("--host", required=True, help="Receiver Host Address")
    parser.add_argument("--port", type=int, required=True, help="Receiver Port Number")
    parser.add_argument("--source_file", required=True, help="Path to the source file to send")
    parser.add_argument("--dest_filename", required=True, help="Destination filename on the receiver side")
    parser.add_argument("--offset", type=int, default=0, help="Offset in bytes to start sending from")
    args = parser.parse_args()

    send_file(
        receiver_host=args.host,
        receiver_port=args.port,
        source_file=args.source_file,
        dest_filename=args.dest_filename,
        offset=args.offset
    )
