import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageEnhance, ImageOps
import matplotlib
# Force matplotlib to use a non-interactive backend to avoid thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime
import uuid
import json

# Import the global variables from globals.py
from globals import log_queue, stop_training, current_batch_size

# Import UI-related functionality from training_ui.py
from training_ui import (
    LoggerCallback, 
    ProgressCallback, 
    AdvancedVisualizationCallback,
    ModelStatsCallback,
    BatchSizeChangeCallback,
    AdaptiveLRScheduler,
    update_stats_chart
)

# Safer way to get entry widget values
def get_entry_value(master, widget_name):
    """Get an integer value from an entry widget with error handling"""
    try:
        # Try to find the widget
        widget = master.nametowidget(widget_name)
        if widget:
            # Try to get and convert its value
            value = widget.get()
            return int(value)
    except (ValueError, KeyError, AttributeError, tk.TclError):
        # Return None if anything goes wrong
        return None
    return None

# Train the fast perceptual loss model
def train_fast_perceptual_model(model, vgg_model, dataset_folder, 
                           batch_size_var, epochs=250, steps_per_epoch=100, 
                           initial_lr=0.005,
                           canvas_frames=None, progress_var=None,
                           stats_canvas=None, stats_fig=None,
                           current_lr_var=None):
    global stop_training, current_batch_size
    stop_training = False
    current_batch_size = int(batch_size_var.get())
    
    # Get root window for thread-safe operations
    root = None
    if hasattr(batch_size_var, 'master'):
        # Get the root window through master attribute
        root = batch_size_var.master.winfo_toplevel()
    elif canvas_frames and len(canvas_frames) > 0:
        root = canvas_frames[0].winfo_toplevel()
        
    # Store UI variables directly on root for access by callbacks
    if root and progress_var:
        root.progress_var = progress_var
    
    log_queue.put("Training FastPerceptualLoss model to mimic VGG19...")
    log_queue.put(f"Using initial learning rate: {initial_lr}")

    # Create a sub-model of VGG19 up to block3_conv3
    vgg_submodel = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block3_conv3').output)
    vgg_submodel.trainable = False

    # Get all JPEG files from the dataset folder
    image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    
    log_queue.put("List of images used for training:")
    for image_file in image_files:
        log_queue.put(f"{image_file}")

    # Prepare training data generator with augmentation using Pillow
    def generate_training_data():
        # Target size for final patches
        target_size = 512
        # Use larger patch size before rotation to avoid black borders
        extraction_size = int(target_size * 1.5)  # 50% larger to allow for rotation
        
        log_queue.put(f"Generating augmented patches of size {target_size}x{target_size} using Pillow")
        
        # Pillow-based augmentation function
        def augment_with_pillow(image_array):
            # Convert numpy array to PIL Image
            # TensorFlow tensor to numpy array
            if isinstance(image_array, tf.Tensor):
                image_array = image_array.numpy()
            
            # Make sure image is in [0, 255] range with uint8 type for PIL
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
                
            pil_image = Image.fromarray(image_array)
            
            # Random flip (50% chance for each direction)
            if np.random.random() > 0.5:
                pil_image = ImageOps.mirror(pil_image)  # Horizontal flip
            if np.random.random() > 0.5:
                pil_image = ImageOps.flip(pil_image)    # Vertical flip
                
            # Random rotation (-45 to 45 degrees)
            if np.random.random() > 0.3:  # 70% chance of rotation
                angle = np.random.uniform(-45, 45)
                pil_image = pil_image.rotate(
                    angle, 
                    resample=Image.BILINEAR,
                    expand=False,         # Don't expand canvas
                    fillcolor=(0, 0, 0)   # Fill with black (will be cropped later)
                )
            
            # Advanced augmentations with higher probability
            # Random color adjustments
            # Brightness adjustment (0.7 to 1.3)
            brightness_factor = np.random.uniform(0.7, 1.3)
            pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness_factor)
            
            # Contrast adjustment (0.7 to 1.3)
            contrast_factor = np.random.uniform(0.7, 1.3)
            pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast_factor)
            
            # Color/Saturation adjustment (0.7 to 1.3)
            color_factor = np.random.uniform(0.7, 1.3)
            pil_image = ImageEnhance.Color(pil_image).enhance(color_factor)
            
            # Convert back to numpy array and normalize to [0, 1] for TensorFlow
            result = np.array(pil_image).astype(np.float32) / 255.0
            
            # Convert to tensor
            return tf.convert_to_tensor(result)
            
        def extract_random_patch(image_tensor):
            # Get image dimensions
            height, width = image_tensor.shape[0], image_tensor.shape[1]
            
            # Only extract patch if image is large enough
            if height > extraction_size and width > extraction_size:
                # Random position to extract patch
                h_start = np.random.randint(0, height - extraction_size)
                w_start = np.random.randint(0, width - extraction_size)
                
                # Extract patch (using numpy slicing for simplicity)
                patch = image_tensor[h_start:h_start+extraction_size, w_start:w_start+extraction_size, :]
            else:
                # If image is too small, resize it to extraction_size with padding
                patch = tf.image.resize_with_pad(image_tensor, extraction_size, extraction_size)
                
            return patch
        
        # MixUp data augmentation function defined inside generate_training_data scope
        def apply_mixup(img1, img2, alpha=0.2):
            # Generate mixup coefficient
            lam = np.random.beta(alpha, alpha)
            # Ensure both images have the same shape
            if img1.shape != img2.shape:
                # Resize the second image to match the first
                img2 = tf.image.resize(img2, (img1.shape[0], img1.shape[1]))
            # Apply mixup
            mixed_img = lam * img1 + (1 - lam) * img2
            return mixed_img
        
        # Cache recently processed images for mixup
        recent_images = []
        max_recent = 10
        
        while True:  # Create an infinite generator
            np.random.shuffle(image_files)  # Shuffle the order of images for each epoch
            for image_file in image_files:
                try:
                    # Read image
                    image_path = os.path.join(dataset_folder, image_file)
                    
                    # Use TensorFlow to read file
                    image_data = tf.io.read_file(image_path)
                    image_tensor = tf.image.decode_image(image_data, channels=3, expand_animations=False)
                    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0  # Normalize to [0, 1]
                    
                    # Extract a larger patch for rotation
                    patch = extract_random_patch(image_tensor)
                    
                    # Apply mixup augmentation with 30% probability
                    if len(recent_images) > 0 and np.random.random() < 0.3:
                        # Select a random image from recent images
                        mix_idx = np.random.randint(0, len(recent_images))
                        # Apply mixup with proper shape handling
                        patch = apply_mixup(patch, recent_images[mix_idx])
                    
                    # Apply Pillow-based augmentations
                    augmented_patch = augment_with_pillow(patch)
                    
                    # Final center crop to target size to eliminate any black borders
                    final_patch = tf.image.resize_with_crop_or_pad(augmented_patch, target_size, target_size)
                    
                    # Store in recent images for future mixup
                    if len(recent_images) >= max_recent:
                        recent_images.pop(0)  # Remove oldest
                    recent_images.append(final_patch)
                    
                    # Extract VGG19 features as targets using the sub-model
                    vgg_features = vgg_submodel(tf.expand_dims(final_patch, axis=0))[0]
                    
                    yield final_patch, vgg_features
                except Exception as e:
                    log_queue.put(f"Error processing {image_file}: {e}")
                    continue  # Skip this image and move to the next one
    
    # Function to create a dataset with a specific batch size
    def create_dataset(batch_size):
        return tf.data.Dataset.from_generator(
            generate_training_data,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 256), dtype=tf.float32)  # Match VGG19 block3_conv3 output
            )
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Compile the model with the specified initial learning rate
    log_queue.put(f"Initializing optimizer with learning rate: {initial_lr}")
    
    optimizer = Adam(
        learning_rate=initial_lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )
    
    # Compile with the optimizer
    model.compile(optimizer=optimizer, loss='mse')

    # Define checkpoint directory
    checkpoint_dir = './Checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define directory for training history
    history_dir = './History'
    os.makedirs(history_dir, exist_ok=True)
    
    # Create a unique filename for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = os.path.join(history_dir, f'training_history_{timestamp}.json')
    
    # Initialize history dictionary for JSON file
    history_data = {
        'epochs': [],
        'loss': [],
        'learning_rate': [],
        'batch_sizes': [],  # Track batch size changes
        'timestamp': timestamp,
        'model_info': {
            'parameters': model.count_params(),
            'initial_lr': initial_lr,
            'initial_batch_size': current_batch_size
        }
    }
    
    # Function to save history to JSON
    def save_history_to_json():
        try:
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=4)
            log_queue.put(f"Training history saved to {history_file}")
        except Exception as e:
            log_queue.put(f"Error saving history: {str(e)}")
    
    # Define checkpoint logic
    checkpoint_filepath = os.path.join(checkpoint_dir, 'fast_perceptual_loss_epoch_{epoch:02d}.h5')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,  # Save the entire model
        save_freq='epoch',        # Save after every epoch
        verbose=1,                # Print messages when saving
        save_best_only=True,      # Only save when the model improves
        monitor='loss'            # Monitor the training loss
    )

    # Check if there are existing checkpoints and load the latest one
    start_epoch = 0
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('fast_perceptual_loss_epoch_')] if os.path.exists(checkpoint_dir) else []
    if checkpoint_files:
        # Extract epoch numbers
        epoch_numbers = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
        latest_epoch = max(epoch_numbers)
        latest_checkpoint = f'fast_perceptual_loss_epoch_{latest_epoch:02d}.h5'
        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        log_queue.put(f"Latest checkpoint found: {latest_checkpoint_path} (Epoch {latest_epoch})")
        
        # Load the model from the latest checkpoint
        model = tf.keras.models.load_model(latest_checkpoint_path)
        start_epoch = latest_epoch
        log_queue.put(f"Training will resume from epoch {start_epoch + 1}")
    else:
        log_queue.put("No checkpoints found. Starting training from scratch.")

    # Create the initial dataset with the current_batch_size
    dataset = create_dataset(current_batch_size)
    log_queue.put(f"Initial batch size set to {current_batch_size}")

    # Use only the first frame for visualization
    viz_frame = canvas_frames[0] if canvas_frames else None
    
    # Create adaptive learning rate scheduler with improved parameters
    adaptive_lr = AdaptiveLRScheduler(
        initial_lr=initial_lr,
        min_lr=1e-6,
        patience=3,                # Reduced patience for quicker response
        reduction_factor=0.7,      # Less aggressive reduction
        aggressive_reduction=0.3,  # More moderate aggressive reduction
        recovery_factor=1.03,     # More gentle recovery
        warmup_epochs=2,           # Shorter warmup
        cooldown=1,                # Shorter cooldown
        verbose=1,                 # Keep verbose mode
        current_lr_var=current_lr_var  # Pass UI variable
    )
    
    # Custom callback for JSON history tracking with more frequent updates
    class HistoryCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            try:
                # Get current learning rate - with error handling
                if hasattr(self.model.optimizer, 'lr'):
                    current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                else:
                    # Fall back to initial learning rate if optimizer is not available
                    current_lr = initial_lr  
                
                # Update history data
                history_data['epochs'].append(epoch)
                history_data['loss'].append(float(logs.get('loss', 0.0)))
                history_data['learning_rate'].append(current_lr)
                history_data['batch_sizes'].append(current_batch_size)  # Track batch size
                
                # Update stats chart more frequently
                if stats_fig and stats_canvas and root:
                    root.after(0, lambda: update_stats_chart(
                        stats_fig, 
                        stats_canvas, 
                        history_data['epochs'],
                        history_data['loss'],
                        history_data['learning_rate']
                    ))
                
                # Save history to JSON file every 2 epochs instead of 5
                if epoch % 2 == 0 or epoch == self.params.get('epochs', epochs) - 1:
                    save_history_to_json()
            except Exception as e:
                log_queue.put(f"Error in history callback: {str(e)}")
                # Still try to update with available data
                save_history_to_json()
    
    # Setup the advanced visualization callback
    visualization_callback = AdvancedVisualizationCallback(
        dataset, vgg_submodel, viz_frame, root, adaptive_lr
    ) if viz_frame and root else None
    
    # Create a more frequent batch-level learning rate update callback
    class LRUpdateCallback(Callback):
        def __init__(self, update_freq=5):
            super(LRUpdateCallback, self).__init__()
            self.update_freq = update_freq
            self.batch_count = 0
            
        def on_batch_end(self, batch, logs=None):
            self.batch_count += 1
            # Update current learning rate display every few batches
            if self.batch_count % self.update_freq == 0 and current_lr_var and root:
                try:
                    # Get current learning rate
                    if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
                        curr_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                        root.after(0, lambda: current_lr_var.set(f"Current LR: {curr_lr:.6f}"))
                except Exception as e:
                    log_queue.put(f"Error updating LR display: {str(e)}")
    
    logger_callback = LoggerCallback()
    stats_callback = ModelStatsCallback()
    batch_size_change_callback = BatchSizeChangeCallback(create_dataset)
    progress_callback = ProgressCallback(root, epochs)
    history_callback = HistoryCallback()
    lr_update_callback = LRUpdateCallback(update_freq=3)  # Update LR display every 3 batches
    
    # Add all callbacks
    callbacks = [
        model_checkpoint_callback,   # Save checkpoints
        logger_callback,             # Log progress
        progress_callback,           # Update progress bar
        stats_callback,              # Model statistics
        batch_size_change_callback,  # Handle batch size changes
        adaptive_lr,                 # Adaptive learning rate scheduler
        history_callback,            # JSON history tracker
        lr_update_callback           # More frequent LR updates
    ]
    
    # Only add visualization if we have a frame
    if visualization_callback:
        callbacks.append(visualization_callback)
    
    # Create a custom training loop that refreshes the dataset between epochs
    current_epoch = start_epoch
    
    # Setup dataset tracking
    active_dataset = dataset
    current_steps = steps_per_epoch
    
    try:
        log_queue.put("Starting training loop with dynamic parameter support...")
        
        while current_epoch < epochs and not stop_training:
            # Update steps_per_epoch if changed via UI
            if hasattr(batch_size_var, 'master'):
                try:
                    # Get updated value from UI - safer access with custom function
                    ui_steps = get_entry_value(batch_size_var.master, '.!panedwindow.!frame.!panedwindow.!frame.!labelframe.!entry2')
                    if ui_steps and ui_steps != current_steps:
                        log_queue.put(f"Steps per epoch changed from {current_steps} to {ui_steps}")
                        current_steps = ui_steps
                except Exception as e:
                    # Just log the error but continue with current value
                    log_queue.put(f"Error reading steps per epoch: {str(e)}")
            
            # Update total epochs if changed via UI
            if hasattr(batch_size_var, 'master'):
                try:
                    # Get updated epochs value from UI
                    ui_epochs = get_entry_value(batch_size_var.master, '.!panedwindow.!frame.!panedwindow.!frame.!labelframe.!entry')
                    if ui_epochs and ui_epochs > epochs:
                        log_queue.put(f"Total epochs increased from {epochs} to {ui_epochs}")
                        epochs = ui_epochs
                        
                        # Update progress callback with new total
                        for callback in callbacks:
                            if isinstance(callback, ProgressCallback):
                                callback.total_epochs = ui_epochs
                except Exception as e:
                    # Just log the error but continue with current value
                    log_queue.put(f"Error reading total epochs: {str(e)}")
            
            # Prepare callbacks dict for this epoch
            callback_logs = {}
            
            # Call on_epoch_begin callbacks
            for callback in callbacks:
                try:
                    callback.on_epoch_begin(current_epoch, callback_logs)
                except Exception as e:
                    log_queue.put(f"Error in callback {callback.__class__.__name__}: {str(e)}")
            
            # Check if batch size callback updated the dataset
            for callback in callbacks:
                if isinstance(callback, BatchSizeChangeCallback) and hasattr(callback, '_data'):
                    try:
                        active_dataset = callback._data
                        log_queue.put(f"New batch size of {current_batch_size} applied for epoch {current_epoch+1}")
                        delattr(callback, '_data')
                    except Exception as e:
                        log_queue.put(f"Error applying new batch size: {str(e)}")
            
            # Train for one epoch with current steps_per_epoch
            try:
                history = model.fit(
                    active_dataset,
                    initial_epoch=current_epoch,
                    epochs=current_epoch+1,
                    steps_per_epoch=current_steps,
                    callbacks=callbacks,
                    verbose=0  # Disable default progress bar
                )
            except Exception as e:
                log_queue.put(f"Error during model.fit: {str(e)}")
                # Try to continue with next epoch
                
            # Increment epoch counter
            current_epoch += 1
            
            # Check if training should stop
            if stop_training:
                log_queue.put("Training stopped by user.")
                break
                
    except Exception as e:
        log_queue.put(f"Error in training loop: {str(e)}")
    finally:
        # Call on_train_end for all callbacks
        for callback in callbacks:
            callback.on_train_end({})

    # Save the final trained model
    final_model_path = os.path.join(checkpoint_dir, 'fast_perceptual_loss_final.h5')
    model.save(final_model_path)
    log_queue.put(f"FastPerceptualLoss model trained and saved to {final_model_path}.")
    
    # Save final history to JSON
    save_history_to_json()
    
    return model