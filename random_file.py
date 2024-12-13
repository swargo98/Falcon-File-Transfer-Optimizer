import os
import random
import string
from datetime import datetime
from tqdm import tqdm

def generate_random_file(file_directory, file_size):
    # Ensure the directory exists
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    # Create a filename with the current timestamp
    random_num = random.randint(0, 1000)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(random_num) + '.txt'
    file_path = os.path.join(file_directory, filename)

    # Generate random content
    with open(file_path, 'w') as f:
        remaining_size = file_size
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Generating file") as pbar:
            while remaining_size > 0:
                chunk_size = min(1024, remaining_size)  # Write in chunks of up to 1KB
                chunk = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + '\n', k=chunk_size))
                f.write(chunk)
                remaining_size -= chunk_size
                pbar.update(chunk_size)

    print(f"Random file generated: {file_path}")

# Example usage
file_directory = "/home/rs75c/Falcon-File-Transfer-Optimizer/src"

# remove existing files
for file in os.listdir(file_directory):
    if file.endswith(".txt"):
        os.remove(os.path.join(file_directory, file))

file_sizes_in_MB = [1, 10, 50, 100]
oneMB = 1024 * 1024

for size in file_sizes_in_MB:
    generate_random_file(file_directory, size * oneMB)
