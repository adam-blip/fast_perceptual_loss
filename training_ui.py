import time
import datetime
import tkinter as tk
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from PIL import Image, ImageTk
import io
import os

# Import the global variables
from globals import log_queue, stop_training, current_batch_size

# Chart and visualization functions
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
            
            # Plot the learning rate - ensure same length arrays
            if len(epochs) != len(lr_values):
                # Trim the longer array to match the shorter one
                length = min(len(epochs), len(lr_values))
                epochs_plot = epochs[:length]
                lr_plot = lr_values[:length]
                log_queue.put(f"Fixed array length mismatch: {len(epochs)} vs {len(lr_values)}")
            else:
                epochs_plot = epochs
                lr_plot = lr_values
                
            ax2.plot(epochs_plot, lr_plot, 'r-')
            ax2.set_title('Learning Rate')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True)
            ax2.set_yscale('log')  # Use log scale for learning rate
            
            # Update the canvas for just this change
            canvas.draw_idle()
    except Exception as e:
        log_queue.put(f"Error updating LR plot: {str(e)}")

# Thread-safe custom callback to display a single sample patch without saving
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
        self.photo_img = None  # Keep reference to prevent garbage collection
        self.last_width = 0
        self.last_height = 0
    
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
        
        # Store current learning rate with improved error handling
        try:
            # Safely get current learning rate
            if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                self.learning_rates.append(current_lr)
            else:
                # Use previous learning rate if available, otherwise use a default
                if self.learning_rates and len(self.learning_rates) > 0:
                    previous_lr = self.learning_rates[-1]
                    self.learning_rates.append(previous_lr)
                else:
                    # Use initial LR as fallback
                    fallback_lr = 0.001  # Default initial LR
                    self.learning_rates.append(fallback_lr)
        except Exception as e:
            log_queue.put(f"Error retrieving learning rate: {str(e)}")
            # Still add a value to maintain array length
            if self.learning_rates and len(self.learning_rates) > 0:
                self.learning_rates.append(self.learning_rates[-1])
            else:
                self.learning_rates.append(0.001)  # Default initial LR

    def on_epoch_end(self, epoch, logs=None):
        if stop_training:
            self.model.stop_training = True
            return
            
        try:
            current_loss = logs.get('loss')
            self.losses.append(current_loss)
            self.epochs.append(epoch)
            
            if not self.sample_data:
                return

            # Get the prediction from the lightweight model
            x, y_true = self.sample_data[0]
            x_input = tf.expand_dims(x, axis=0)
            
            try:
                # Wrap in try/except in case model prediction fails
                y_pred = self.model(x_input)[0]
            except Exception as e:
                log_queue.put(f"Error during model prediction: {str(e)}")
                return
            
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
            
            # Instead of saving to file, render directly to memory
            # Create a BytesIO object to hold the image data
            buf = io.BytesIO()
            patches_fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            buf.seek(0)
            
            # Update the UI directly with this image
            self.root.after(0, lambda: self.update_canvas_from_buffer(buf))
            plt.close(patches_fig)
            
            # Update learning rate plot frequently
            if hasattr(self.root, 'stats_fig') and hasattr(self.root, 'stats_canvas'):
                # Ensure arrays are the same length before updating
                epochs_to_use = self.epochs
                lr_to_use = self.learning_rates[:len(epochs_to_use)]
                
                self.root.after(0, lambda: update_lr_plot(
                    self.root.stats_fig,
                    self.root.stats_canvas,
                    epochs_to_use,
                    self.losses[:len(epochs_to_use)],
                    lr_to_use
                ))
        except Exception as e:
            log_queue.put(f"Error in visualization callback: {str(e)}")
    
    def update_canvas_from_buffer(self, buf):
        """Update the canvas directly from a buffer"""
        try:
            # Clear the frame
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            
            # Open the image from buffer
            img = Image.open(buf)
            
            # Create a label to display the image
            label = tk.Label(self.canvas_frame)
            label.pack(fill=tk.BOTH, expand=True)
            
            # Make the visualization responsive to window resizing
            self.canvas_frame.update_idletasks()  # Force layout update
            
            # Calculate the available space
            frame_width = self.canvas_frame.winfo_width()
            frame_height = self.canvas_frame.winfo_height()
            
            # Calculate scaling factor to fit the image within the frame while maintaining aspect ratio
            img_width, img_height = img.size
            width_ratio = frame_width / img_width
            height_ratio = frame_height / img_height
            scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale if image is smaller
            
            # Calculate new dimensions
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            
            # Resize the image to fit the frame
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage and display
            self.photo_img = ImageTk.PhotoImage(resized_img)
            label.config(image=self.photo_img)
            
            # Add resize binding to the canvas frame
            self.canvas_frame.bind("<Configure>", self.on_frame_resize)
            
            # Store original image and buffer for resize
            self.original_img = img
            self.img_buffer = buf
            
        except Exception as e:
            log_queue.put(f"Error updating visualization: {str(e)}")
    
    def on_frame_resize(self, event=None):
        """Handle frame resize events to scale the visualization properly"""
        if hasattr(self, 'original_img') and self.original_img:
            try:
                # Calculate the new available space
                frame_width = self.canvas_frame.winfo_width() 
                frame_height = self.canvas_frame.winfo_height()
                
                # Get original image dimensions
                img_width, img_height = self.original_img.size
                
                # Calculate new scaling factor
                width_ratio = frame_width / img_width
                height_ratio = frame_height / img_height
                scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale if image is smaller
                
                # Calculate new dimensions
                new_width = int(img_width * scale_factor)
                new_height = int(img_height * scale_factor)
                
                # Only resize if dimensions changed significantly
                if abs(new_width - getattr(self, 'last_width', 0)) > 10 or abs(new_height - getattr(self, 'last_height', 0)) > 10:
                    # Resize the image to fit the frame
                    resized_img = self.original_img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Find the label widget
                    for widget in self.canvas_frame.winfo_children():
                        if isinstance(widget, tk.Label):
                            # Update the PhotoImage
                            self.photo_img = ImageTk.PhotoImage(resized_img)
                            widget.config(image=self.photo_img)
                            
                            # Store current dimensions
                            self.last_width = new_width
                            self.last_height = new_height
                            break
            except Exception as e:
                log_queue.put(f"Error resizing visualization: {str(e)}")