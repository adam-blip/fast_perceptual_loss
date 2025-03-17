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
import random
import string

# Import the global variables from globals.py
from globals import log_queue, stop_training, current_batch_size, default_epochs, default_steps_per_epoch, default_learning_rate, default_image_size

# Import custom layers from model.py
from model import WeightedAddLayer, MeanReduceLayer, MaxReduceLayer, create_fast_perceptual_model

# Import UI-related functionality from both files
from training_ui import update_stats_chart, AdvancedVisualizationCallback
from training_helper import (
    LoggerCallback, 
    ProgressCallback, 
    ModelStatsCallback,
    BatchSizeChangeCallback
)
from training_scheduler import ImprovedAdaptiveLRScheduler

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
    except (ValueError, KeyError, AttributeError, tf.errors.OpError):
        # Return None if anything goes wrong
        return None
    return None

# Train the fast perceptual loss model
def train_fast_perceptual_model(model, vgg_model, dataset_folder, 
                           batch_size_var, epochs=default_epochs, steps_per_epoch=default_steps_per_epoch, 
                           initial_lr=default_learning_rate,
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
    
    # Print VGG model dimensions to understand downsampling ratio
    log_queue.put(f"VGG19 Submodel: Input shape {vgg_model.input.shape}, Output shape {vgg_submodel.output.shape}")

    # Get all JPEG files from the dataset folder
    image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    
    log_queue.put("List of images used for training:")
    for image_file in image_files:
        log_queue.put(f"{image_file}")

    # Prepare training data generator with augmentation using Pillow
    def generate_training_data():
        # Target size for final patches
        target_size = default_image_size
        # Use larger patch size before rotation to avoid black borders
        extraction_size = int(target_size * 1.5)  # 50% larger to allow for rotation
        
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
                # Force resize to square dimensions before padding
                min_dim = min(height, width)
                patch = tf.image.resize_with_crop_or_pad(image_tensor, min_dim, min_dim)
                patch = tf.image.resize(patch, [extraction_size, extraction_size])
            
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
                    # Use the SAME input for both models
                    vgg_features = vgg_submodel(tf.expand_dims(final_patch, axis=0))[0]
                    
                    yield final_patch, vgg_features
                except Exception as e:
                    log_queue.put(f"Error processing {image_file}: {e}")
                    continue  # Skip this image and move to the next one
    
    # Function to create a dataset with a specific batch size - FIXED FOR HDF5 ERRORS
    def create_dataset(batch_size):
        # Generate a truly unique name with timestamp and random component to avoid conflicts
        timestamp = int(time.time())
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        unique_name = f"dataset_{timestamp}_{random_suffix}"
        
        # Create the dataset with the guaranteed unique name
        dataset = tf.data.Dataset.from_generator(
            generate_training_data,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 256), dtype=tf.float32)  # Match VGG19 block3_conv3 output
            ),
            name=unique_name
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
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
    
    # Force optimizer initialization with a dummy batch to avoid warnings
    try:
        # Create dummy data with dimensions that match the target size/4
        input_size = default_image_size
        output_size = input_size // 4  # Expected output size (128x128)
        
        dummy_x = tf.zeros((1, input_size, input_size, 3))
        dummy_y = tf.zeros((1, output_size, output_size, 256))
        
        # Run one training step to initialize optimizer
        model.train_on_batch(dummy_x, dummy_y)
        log_queue.put(f"Optimizer initialized with dummy batch")
    except Exception as e:
        log_queue.put(f"Error during optimizer initialization: {str(e)}")

    # Define checkpoint directory
    checkpoint_dir = './Checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define checkpoint logic with unique file naming to avoid HDF5 conflicts
    timestamp = int(time.time())
    checkpoint_filepath = os.path.join(
        checkpoint_dir, 
        f'fast_perceptual_loss_epoch_{{epoch:02d}}_{timestamp}.h5'
    )
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
        epoch_numbers = []
        for f in checkpoint_files:
            try:
                # Handle both naming formats (with and without timestamp)
                if '_' in f.split('epoch_')[1]:
                    epoch_num = int(f.split('epoch_')[1].split('_')[0])
                else:
                    epoch_num = int(f.split('epoch_')[1].split('.')[0])
                epoch_numbers.append(epoch_num)
            except (IndexError, ValueError):
                continue
                
        if epoch_numbers:
            latest_epoch = max(epoch_numbers)
            # Find the exact filename with this epoch number (could have different timestamps)
            matching_files = [f for f in checkpoint_files if f'epoch_{latest_epoch:02d}' in f]
            if matching_files:
                latest_checkpoint = matching_files[0]  # Use the first matching file
                latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                log_queue.put(f"Latest checkpoint found: {latest_checkpoint_path} (Epoch {latest_epoch})")
                
                # Register custom objects before loading the model
                custom_objects = {
                    'WeightedAddLayer': WeightedAddLayer,
                    'MeanReduceLayer': MeanReduceLayer,
                    'MaxReduceLayer': MaxReduceLayer
                }
                
                try:
                    # Load the model from the latest checkpoint with custom objects
                    model = tf.keras.models.load_model(latest_checkpoint_path, custom_objects=custom_objects)
                    start_epoch = latest_epoch
                    log_queue.put(f"Training will resume from epoch {start_epoch + 1}")
                    
                    # Re-compile the model after loading to ensure optimizer is initialized
                    model.compile(optimizer=optimizer, loss='mse')
                    
                    # Re-initialize optimizer with a dummy batch
                    try:
                        # Create dummy data with dimensions that match the target size/4
                        input_size = default_image_size
                        output_size = input_size // 4  # Expected output size
                        
                        dummy_x = tf.zeros((1, input_size, input_size, 3))
                        dummy_y = tf.zeros((1, output_size, output_size, 256))
                        
                        # Run one training step to initialize optimizer
                        model.train_on_batch(dummy_x, dummy_y)
                        log_queue.put("Optimizer re-initialized successfully after loading checkpoint")
                    except Exception as e:
                        log_queue.put(f"Error during optimizer re-initialization: {str(e)}")
                except Exception as e:
                    log_queue.put(f"Error loading checkpoint: {str(e)}, starting from scratch")
        else:
            log_queue.put("No valid checkpoints found. Starting training from scratch.")
    else:
        log_queue.put("No checkpoints found. Starting training from scratch.")

    # Create the initial dataset with the current_batch_size
    dataset = create_dataset(current_batch_size)
    log_queue.put(f"Initial batch size set to {current_batch_size}")

    # Use only the first frame for visualization
    viz_frame = canvas_frames[0] if canvas_frames else None
    
    # Create improved adaptive learning rate scheduler with enhanced parameters
    adaptive_lr = ImprovedAdaptiveLRScheduler(
        initial_lr=initial_lr,
        min_lr=1e-7,               # Lower min LR for finer control
        patience=2,                # More responsive patience
        reduction_factor=0.4,      # More aggressive reduction
        aggressive_reduction=0.3,  # Keep moderate aggressive reduction
        recovery_factor=1.05,      # Keep gentle recovery
        warmup_epochs=2,           # Keep shorter warmup
        cooldown=1,                # Keep shorter cooldown
        verbose=1,                 # Keep verbose mode
        current_lr_var=current_lr_var,  # Pass UI variable
        early_reaction_threshold=0.03,  # Detect trends early
        loss_memory_factor=0.7,         # Smooth loss tracking
        trend_detection_window=3        # How many epochs to consider for trend
    )
    
    # Modified DataTrackingCallback to track and update charts but not save history
    class DataTrackingCallback(Callback):
        def __init__(self):
            super(DataTrackingCallback, self).__init__()
            self.epochs = []
            self.losses = []
            self.learning_rates = []
            
        def on_epoch_end(self, epoch, logs=None):
            try:
                # Get current learning rate - with error handling
                if hasattr(self.model.optimizer, 'lr'):
                    current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                else:
                    # Fall back to initial learning rate if optimizer is not available
                    current_lr = initial_lr  
                
                # Update tracked data
                self.epochs.append(epoch)
                self.losses.append(float(logs.get('loss', 0.0)))
                self.learning_rates.append(current_lr)
                
                # Update stats chart more frequently
                if stats_fig and stats_canvas and root:
                    root.after(0, lambda: update_stats_chart(
                        stats_fig, 
                        stats_canvas, 
                        self.epochs,
                        self.losses,
                        self.learning_rates
                    ))
            except Exception as e:
                log_queue.put(f"Error in data tracking callback: {str(e)}")
    
    # Setup the advanced visualization callback
    visualization_callback = AdvancedVisualizationCallback(
        dataset, vgg_submodel, viz_frame, root, adaptive_lr
    ) if viz_frame and root else None
    
    logger_callback = LoggerCallback()
    stats_callback = ModelStatsCallback()
    batch_size_change_callback = BatchSizeChangeCallback(create_dataset)
    progress_callback = ProgressCallback(root, epochs)
    data_tracking_callback = DataTrackingCallback()
    
    # Add all callbacks
    callbacks = [
        model_checkpoint_callback,    # Save checkpoints
        logger_callback,              # Log progress
        progress_callback,            # Update progress bar
        stats_callback,               # Model statistics
        batch_size_change_callback,   # Handle batch size changes
        adaptive_lr,                  # Adaptive learning rate scheduler
        data_tracking_callback,       # Data tracking for charts
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
                # Try to recover from dataset error
                if "name already exists" in str(e):
                    log_queue.put("Attempting to recover from dataset naming conflict...")
                    # Create a new dataset with a definitely unique name
                    try:
                        active_dataset = create_dataset(current_batch_size)
                        log_queue.put("Successfully recreated dataset with unique name")
                        # Continue with next epoch
                        current_epoch += 1
                        continue
                    except Exception as recovery_error:
                        log_queue.put(f"Failed to recover: {str(recovery_error)}")
                        break
                else:
                    # For other errors, try to continue with next epoch
                    log_queue.put(f"Will attempt to continue with next epoch")
                    current_epoch += 1
                    continue
                
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
            try:
                callback.on_train_end({})
            except Exception as e:
                log_queue.put(f"Error in callback on_train_end: {str(e)}")

    # Save the final trained model with unique timestamp to avoid conflicts
    final_timestamp = int(time.time())
    final_model_path = os.path.join(checkpoint_dir, f'fast_perceptual_loss_final_{final_timestamp}.h5')
    try:
        model.save(final_model_path)
        log_queue.put(f"FastPerceptualLoss model trained and saved to {final_model_path}.")
    except Exception as e:
        log_queue.put(f"Error saving final model: {str(e)}")
        # Try alternate save methods if the first fails
        try:
            log_queue.put("Attempting to save with different method...")
            model.save_weights(os.path.join(checkpoint_dir, f'fast_perceptual_loss_weights_{final_timestamp}.h5'))
            log_queue.put(f"Model weights saved successfully.")
        except Exception as e2:
            log_queue.put(f"Failed to save weights as well: {str(e2)}")
    
    return model