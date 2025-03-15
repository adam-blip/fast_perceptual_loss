# globals.py
import queue
import os

# Global variables
stop_training = False
log_queue = queue.Queue()
current_batch_size = 2  # Starting batch size is fixed at 2

# Ensure directories exist
os.makedirs('./Checkpoints', exist_ok=True)
os.makedirs('./History', exist_ok=True)
os.makedirs('./Visualizations', exist_ok=True)