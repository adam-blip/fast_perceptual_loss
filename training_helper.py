import time
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

# Import the global variables
from globals import log_queue, stop_training, current_batch_size

# Import the scheduler from the separate file
from training_scheduler import ImprovedAdaptiveLRScheduler, AdaptiveLRScheduler

# Custom callback for batch size change
class BatchSizeChangeCallback(Callback):
    def __init__(self, dataset_gen_func):
        super(BatchSizeChangeCallback, self).__init__()
        self.dataset_gen_func = dataset_gen_func
        self.last_batch_size = current_batch_size
        
    def on_epoch_end(self, epoch, logs=None):
        global current_batch_size
        
        # Check if batch size has changed
        if self.last_batch_size != current_batch_size:
            log_queue.put(f"Batch size changed from {self.last_batch_size} to {current_batch_size}")
            
            # Create a new dataset with the updated batch size
            try:
                new_dataset = self.dataset_gen_func(current_batch_size)
                
                # Store the new dataset for the next fit call
                self._data = new_dataset
                
                # Notify user about the change
                log_queue.put(f"Dataset updated with new batch size: {current_batch_size}")
            except Exception as e:
                log_queue.put(f"Error creating dataset with new batch size: {str(e)}")
            
        # Save the current batch size for the next epoch
        self.last_batch_size = current_batch_size
        
    def on_epoch_begin(self, epoch, logs=None):
        # Apply the new dataset at the beginning of an epoch if it exists
        if hasattr(self, '_data'):
            try:
                self.model._dataset = self._data
                log_queue.put(f"Applying new batch size {current_batch_size} for epoch {epoch+1}")
                delattr(self, '_data')  # Clear the stored dataset after applying
            except Exception as e:
                log_queue.put(f"Error applying new dataset: {str(e)}")

# Custom logger callback
class LoggerCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # Log every 10 batches
            try:
                log_queue.put(f"Step {batch} - Loss: {logs.get('loss'):.6f}")
            except Exception as e:
                log_queue.put(f"Error in logger callback: {str(e)}")
        
        if stop_training:
            self.model.stop_training = True

# Summary stats callback for model monitoring
class ModelStatsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:  # Every 10 epochs
            try:
                # Calculate and log model metrics
                log_queue.put(f"Model metrics at epoch {epoch}:")
                
                # Calculate parameter count
                trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
                non_trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
                
                log_queue.put(f"  Trainable params: {trainable_params:,}")
                log_queue.put(f"  Total params: {trainable_params + non_trainable_params:,}")
                log_queue.put(f"  Current loss: {logs.get('loss'):.6f}")
            except Exception as e:
                log_queue.put(f"Error in model stats callback: {str(e)}")

# Progress callback to update the progress bar, percentage and ETA
class ProgressCallback(Callback):
    def __init__(self, root, total_epochs):
        super(ProgressCallback, self).__init__()
        self.root = root
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_times = []
        
    def on_epoch_begin(self, epoch, logs=None):
        if not self.root:
            return
            
        try:
            # Get progress value
            progress_value = (epoch) / self.total_epochs * 100
            
            # Direct use of progress_var if passed via train_fast_perceptual_model 
            if hasattr(self.root, 'progress_var') and self.root.progress_var:
                self.root.after(0, lambda: self.root.progress_var.set(progress_value))
            
            # Direct use of UI elements if passed through
            if hasattr(self.root, 'percentage_var') and self.root.percentage_var:
                self.root.after(0, lambda: self.root.percentage_var.set(f"{progress_value:.1f}%"))
                
            if epoch > 0 and hasattr(self.root, 'eta_var') and self.root.eta_var:
                # Calculate average epoch time
                if len(self.epoch_times) > 0:
                    avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                    remaining_epochs = self.total_epochs - (epoch)
                    estimated_seconds = avg_epoch_time * remaining_epochs
                    
                    # Format ETA as hours:minutes:seconds
                    eta_str = str(datetime.timedelta(seconds=int(estimated_seconds)))
                    self.root.after(0, lambda: self.root.eta_var.set(f"ETA: {eta_str}"))
        except Exception as e:
            log_queue.put(f"Error updating progress: {str(e)}")
    
    def on_epoch_end(self, epoch, logs=None):
        # Record epoch time for ETA calculation
        if epoch == 0:
            self.start_time = time.time()  # Reset start time after first epoch
        else:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            self.start_time = time.time()  # Reset for next epoch
            
        # Update progress again at end of epoch
        if self.root:
            progress_value = (epoch + 1) / self.total_epochs * 100
            if hasattr(self.root, 'progress_var') and self.root.progress_var:
                self.root.after(0, lambda: self.root.progress_var.set(progress_value))
            if hasattr(self.root, 'percentage_var') and self.root.percentage_var:
                self.root.after(0, lambda: self.root.percentage_var.set(f"{progress_value:.1f}%"))
            
    def on_train_end(self, logs=None):
        log_queue.put("Training finished!")