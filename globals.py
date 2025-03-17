# globals.py
import queue
import os

# Global variables
stop_training = False
log_queue = queue.Queue()

# Training parameters
current_batch_size = 32
default_epochs = 100
default_steps_per_epoch = 25
default_learning_rate = 0.001
default_image_size = 256

# Ensure directories exist
os.makedirs('./Checkpoints', exist_ok=True)