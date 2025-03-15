import time
import datetime
import tkinter as tk
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from PIL import Image, ImageTk
import uuid
import json
import os

# Import the global variables
from globals import log_queue, stop_training, current_batch_size

# Update stats chart - now just loss and learning rate
def update_stats_chart(fig, canvas, epochs, loss_values, lr_values):
    """Update the stats chart with current training data"""
    try:
        fig.clear()
        
        # Create loss plot
        ax1 = fig.add_subplot(211)  # Loss plot
        ax1.plot(epochs, loss_values, 'b-')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Create learning rate plot
        ax2 = fig.add_subplot(212)  # Learning rate plot
        ax2.plot(epochs, lr_values, 'r-')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
        ax2.set_yscale('log')  # Use log scale for learning rate
        
        # Adjust layout
        fig.tight_layout()
        
        # Update the canvas
        canvas.draw_idle()
    except Exception as e:
        log_queue.put(f"Error updating stats chart: {str(e)}")

# Function specifically for LR updates - called more frequently
def update_lr_plot(fig, canvas, epochs, loss_values, lr_values):
    """Update just the learning rate plot for more frequent updates"""
    try:
        # Use the existing figure structure
        if len(fig.axes) > 1:
            ax2 = fig.axes[1]
            ax2.clear()
            
            # Plot the learning rate
            ax2.plot(epochs, lr_values, 'r-')
            ax2.set_title('Learning Rate')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True)
            ax2.set_yscale('log')  # Use log scale for learning rate
            
            # Update the canvas for just this change
            canvas.draw_idle()
    except Exception as e:
        log_queue.put(f"Error updating LR plot: {str(e)}")

# Advanced adaptive learning rate scheduler
class AdaptiveLRScheduler(Callback):
    def __init__(self, initial_lr=0.005, min_lr=1e-6, patience=3, 
                 reduction_factor=0.5, warmup_epochs=3, cooldown=2,
                 aggressive_reduction=0.2, recovery_factor=1.2, verbose=1,
                 current_lr_var=None):
        super(AdaptiveLRScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.reduction_factor = reduction_factor
        self.aggressive_reduction = aggressive_reduction
        self.recovery_factor = recovery_factor
        self.warmup_epochs = warmup_epochs
        self.cooldown = cooldown
        self.verbose = verbose
        self.wait = 0
        self.cooldown_counter = 0
        self.best_loss = float('inf')
        self.history = {'lr': [], 'loss': []}
        self.current_lr_var = current_lr_var
        
    def on_train_begin(self, logs=None):
        # Initialize model optimizer explicitly to avoid 'NoneType' errors
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            log_queue.put("Warning: Optimizer not found in model. Initializing optimizer...")
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_lr),
                loss='mse'
            )
        
        # Verify optimizer has been initialized
        if hasattr(self.model, 'optimizer'):
            log_queue.put(f"Optimizer initialized successfully: {type(self.model.optimizer).__name__}")
            # Set initial learning rate explicitly
            tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)
            # Update UI immediately
            if self.current_lr_var:
                try:
                    self.current_lr_var.set(f"Current LR: {self.initial_lr:.6f}")
                except Exception as e:
                    log_queue.put(f"Error updating LR display: {str(e)}")
        else:
            log_queue.put("Warning: Failed to initialize optimizer!")
        
    def on_epoch_begin(self, epoch, logs=None):
        # Safety check for optimizer existence
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            log_queue.put("Warning: Optimizer not available at epoch begin!")
            return
            
        # Warmup phase
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            
            if self.verbose > 0:
                log_queue.put(f"Epoch {epoch+1}: Warmup phase, LR set to {lr:.6f}")
        else:
            # Get current learning rate
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            # Just log the current rate
            if self.verbose > 0:
                log_queue.put(f"Epoch {epoch+1}: Current LR is {lr:.6f}")
        
        # Store learning rate for history
        self.history['lr'].append(lr)
        
        # Update the UI variable with current learning rate
        if self.current_lr_var:
            try:
                self.current_lr_var.set(f"Current LR: {lr:.6f}")
            except Exception as e:
                log_queue.put(f"Error updating LR display: {str(e)}")
        
    def on_epoch_end(self, epoch, logs=None):
        # Safety check for optimizer existence
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            log_queue.put("Warning: Optimizer not available at epoch end!")
            return
            
        current_loss = logs.get('loss')
        self.history['loss'].append(current_loss)
        
        # Skip LR adjustments during warmup
        if epoch < self.warmup_epochs:
            return
        
        # Get current learning rate
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
        # If we're in cooldown period, just decrease counter and return
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        # Check if loss improved
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            
            # If we've had at least 5 epochs and loss is consistently decreasing
            if len(self.history['loss']) >= 5 and all(self.history['loss'][-i] > self.history['loss'][-i+1] for i in range(5, 1, -1)):
                # Try slightly increasing the learning rate if it's been decreasing steadily
                new_lr = min(current_lr * self.recovery_factor, self.initial_lr)
                if new_lr > current_lr and self.verbose > 0:
                    log_queue.put(f"Loss steadily decreasing, slightly increasing LR to {new_lr:.6f}")
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    
                    # Update UI immediately
                    if self.current_lr_var:
                        try:
                            self.current_lr_var.set(f"Current LR: {new_lr:.6f}")
                        except Exception as e:
                            log_queue.put(f"Error updating LR display: {str(e)}")
        else:
            self.wait += 1
            
            # Check for significant loss increase
            if len(self.history['loss']) >= 2:
                loss_ratio = current_loss / self.history['loss'][-2]
                
                # If loss increased significantly, reduce LR more aggressively
                if loss_ratio > 1.1:  # 10% increase in loss
                    new_lr = max(current_lr * self.aggressive_reduction, self.min_lr)
                    if self.verbose > 0:
                        log_queue.put(f"Significant loss increase detected ({loss_ratio:.2f}x), aggressively reducing LR to {new_lr:.6f}")
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    
                    # Update UI immediately
                    if self.current_lr_var:
                        try:
                            self.current_lr_var.set(f"Current LR: {new_lr:.6f}")
                        except Exception as e:
                            log_queue.put(f"Error updating LR display: {str(e)}")
                            
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                    return
            
            # If patience is reached, reduce LR normally
            if self.wait >= self.patience:
                new_lr = max(current_lr * self.reduction_factor, self.min_lr)
                if self.verbose > 0:
                    log_queue.put(f"Patience reached, reducing LR to {new_lr:.6f}")
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                
                # Update UI immediately
                if self.current_lr_var:
                    try:
                        self.current_lr_var.set(f"Current LR: {new_lr:.6f}")
                    except Exception as e:
                        log_queue.put(f"Error updating LR display: {str(e)}")
                        
                self.cooldown_counter = self.cooldown
                self.wait = 0

# Custom callback for batch size change
class BatchSizeChangeCallback(Callback):
    def __init__(self, dataset_gen_func):
        super(BatchSizeChangeCallback, self).__init__()
        self.dataset_gen_func = dataset_gen_func
        self.last_batch_size = current_batch_size
        
    def on_epoch_end(self, epoch, logs=None):
        global current_batch_size
        
        # Log the current batch size
        log_queue.put(f"Current batch size: {current_batch_size}")
        
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
                log_queue.put(f"Batch {batch} - Loss: {logs.get('loss'):.6f}")
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
                log_queue.put(f"  Non-trainable params: {non_trainable_params:,}")
                log_queue.put(f"  Total params: {trainable_params + non_trainable_params:,}")
                log_queue.put(f"  Current loss: {logs.get('loss'):.6f}")
            except Exception as e:
                log_queue.put(f"Error in model stats callback: {str(e)}")

# Thread-safe custom callback to display a single sample patch with LR/loss diagram
class AdvancedVisualizationCallback(Callback):
    def __init__(self, dataset, vgg_submodel, canvas_frame, root, adaptive_lr_scheduler=None):
        super(AdvancedVisualizationCallback, self).__init__()
        self.dataset = dataset
        self.vgg_submodel = vgg_submodel
        self.canvas_frame = canvas_frame
        self.root = root  # Store reference to root for thread-safe operations
        self.adaptive_lr_scheduler = adaptive_lr_scheduler
        self.sample_data = None
        self.losses = []
        self.learning_rates = []
        self.epochs = []
    
    def on_train_begin(self, logs=None):
        # Make canvas frame destroy_widgets method available
        if self.canvas_frame:
            self.canvas_frame.destroy_widgets = self.destroy_widgets
    
    def destroy_widgets(self):
        """Clear all widgets from the canvas frame"""
        try:
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
        except Exception as e:
            log_queue.put(f"Error clearing canvas: {str(e)}")
        
    def on_epoch_begin(self, epoch, logs=None):
        # Get sample data once at the beginning
        if self.sample_data is None:
            try:
                self.sample_data = []
                for x_batch, y_batch in self.dataset.take(1):
                    # Only take the first sample from the batch
                    self.sample_data.append((x_batch[0], y_batch[0]))
            except Exception as e:
                log_queue.put(f"Error getting sample data: {str(e)}")
                return
        
        # Store current learning rate
        try:
            # Safely get current learning rate
            if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                self.learning_rates.append(current_lr)
            else:
                log_queue.put("Warning: Unable to get current learning rate in visualization callback")
        except Exception as e:
            log_queue.put(f"Error storing learning rate: {str(e)}")

    def on_epoch_end(self, epoch, logs=None):
        if stop_training:
            self.model.stop_training = True
            return
            
        try:
            current_loss = logs.get('loss')
            self.losses.append(current_loss)
            self.epochs.append(epoch)
            
            log_queue.put(f"Epoch {epoch+1} completed. Loss: {current_loss:.6f}")
            
            if not self.sample_data:
                return

            # Get the prediction from the lightweight model
            x, y_true = self.sample_data[0]
            x_input = tf.expand_dims(x, axis=0)
            y_pred = self.model(x_input)[0]
            
            # Get current learning rate
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            
            # PATCH VISUALIZATION INCLUDING FEATURE DIFFERENCE
            patches_fig = plt.figure(figsize=(10, 3))  # Wider and shorter figure for horizontal layout
            
            # 1x4 layout for patches and feature diff in a single row
            # Original image
            ax1 = patches_fig.add_subplot(141)
            ax1.imshow(x.numpy())
            ax1.set_title("Input Image")
            ax1.axis('off')
            
            # Feature visualization for target VGG
            ax2 = patches_fig.add_subplot(142)
            target_viz = tf.reduce_sum(y_true, axis=-1).numpy()
            target_viz = (target_viz - target_viz.min()) / (target_viz.max() - target_viz.min() + 1e-7)
            ax2.imshow(target_viz, cmap='viridis')
            ax2.set_title("Target Features (VGG)")
            ax2.axis('off')
            
            # Feature visualization for prediction
            ax3 = patches_fig.add_subplot(143)
            pred_viz = tf.reduce_sum(y_pred, axis=-1).numpy()
            pred_viz = (pred_viz - pred_viz.min()) / (pred_viz.max() - pred_viz.min() + 1e-7)
            ax3.imshow(pred_viz, cmap='viridis')
            ax3.set_title("Predicted Features")
            ax3.axis('off')
            
            # Feature difference
            ax4 = patches_fig.add_subplot(144)
            diff_viz = tf.abs(y_pred - y_true)
            diff_viz = tf.reduce_sum(diff_viz, axis=-1).numpy()
            diff_viz = (diff_viz - diff_viz.min()) / (diff_viz.max() - diff_viz.min() + 1e-7)
            im = ax4.imshow(diff_viz, cmap='hot')
            ax4.set_title(f"Feature Difference (Epoch {epoch+1})")
            ax4.axis('off')
            patches_fig.colorbar(im, ax=ax4, orientation='vertical', fraction=0.046, pad=0.04)
            
            # Adjust layout for patch visualization
            patches_fig.tight_layout()
            
            # Save the figure with a unique filename
            unique_id = str(uuid.uuid4())[:8]  # Short UUID for filename
            patches_file = f'vis_patches_{unique_id}_epoch_{epoch+1}.png'
            patches_fig.savefig(patches_file, bbox_inches='tight', dpi=120)
            plt.close(patches_fig)
            
            # Update the patch visualization in the main thread
            self.root.after(0, self.update_canvas, patches_file)
            
            # Update learning rate plot frequently
            # Update every epoch instead of waiting for HistoryCallback
            if hasattr(self.root, 'stats_fig') and hasattr(self.root, 'stats_canvas'):
                self.root.after(0, lambda: update_lr_plot(
                    self.root.stats_fig,
                    self.root.stats_canvas,
                    self.epochs,
                    self.losses,
                    self.learning_rates
                ))
        except Exception as e:
            log_queue.put(f"Error in visualization callback: {str(e)}")
        
    def update_canvas(self, temp_file):
        """
        Update the canvas in the main thread
        """
        try:
            # Import PIL.ImageTk here to ensure it's available
            from PIL import ImageTk
            
            # Clear the frame
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            
            # Open the saved image with PIL (safer for threading)
            img = Image.open(temp_file)
            
            # Create a label to display the image
            label = tk.Label(self.canvas_frame)
            label.pack(fill=tk.BOTH, expand=True)
            
            # Calculate the available space
            self.canvas_frame.update()
            frame_width = self.canvas_frame.winfo_width()
            frame_height = self.canvas_frame.winfo_height()
            
            # Calculate scaling factor to fit the image within the frame while maintaining aspect ratio
            img_width, img_height = img.size
            width_ratio = frame_width / img_width
            height_ratio = frame_height / img_height
            scale_factor = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            
            # Resize the image to fit the frame
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage and display
            photo_img = ImageTk.PhotoImage(resized_img)
            label.config(image=photo_img)
            label.image = photo_img  # Keep reference to prevent garbage collection
            
            # Keep the file for later reference (don't delete)
            # This allows reviewing all visualizations after training
            log_queue.put(f"Visualization saved to {temp_file}")
        except Exception as e:
            log_queue.put(f"Error updating visualization: {str(e)}")

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