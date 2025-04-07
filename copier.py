import shutil
import os

# List of source files and their destination directories
files_to_copy = [
    ("models/training_dicrete_w_history_minibatch_mlp_deepseek_v9_policy_400000.pth", "finals/"),
    ("models/training_dicrete_w_history_minibatch_mlp_deepseek_v9_value_400000.pth", "finals/"),
    ("models/training_dicrete_w_history_minibatch_mlp_deepseek_v10_policy_400000.pth", "finals/"),
    ("models/training_dicrete_w_history_minibatch_mlp_deepseek_v10_value_400000.pth", "finals/"),
    ("models/training_dicrete_w_history_minibatch_mlp_deepseek_v11_policy_400000.pth", "finals/"),
    ("models/training_dicrete_w_history_minibatch_mlp_deepseek_v11_value_400000.pth", "finals/"),
    ("models/training_dicrete_w_history_minibatch_mlp_deepseek_v12_policy_400000.pth", "finals/"),
    ("models/training_dicrete_w_history_minibatch_mlp_deepseek_v12_value_400000.pth", "finals/")
]

# Ensure destination directories exist and copy files
for source, destination in files_to_copy:
    try:
        # Create destination directory if it does not exist
        os.makedirs(destination, exist_ok=True)

        # Construct the full destination file path
        destination_file = os.path.join(destination, os.path.basename(source))

        # print file sizes
        print(f"Source file size: {os.path.getsize(source)}")

        # Copy the file
        shutil.copy(source, destination_file)
        print(f"Copied {source} to {destination_file}")
        print(f"Destination file size: {os.path.getsize(destination_file)}")
    except Exception as e:
        print(f"Error copying {source}: {e}")
