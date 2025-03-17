import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# Import the global variables
from globals import log_queue

# Improved adaptive learning rate scheduler
class ImprovedAdaptiveLRScheduler(Callback):
    def __init__(self, initial_lr=0.001, min_lr=1e-7, patience=2, 
                 reduction_factor=0.5, warmup_epochs=2, cooldown=1,
                 aggressive_reduction=0.3, recovery_factor=1.05, verbose=1,
                 current_lr_var=None, early_reaction_threshold=0.03,
                 loss_memory_factor=0.7, trend_detection_window=3):
        super(ImprovedAdaptiveLRScheduler, self).__init__()
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
        self.early_reaction_threshold = early_reaction_threshold
        self.loss_memory_factor = loss_memory_factor
        self.trend_detection_window = trend_detection_window
        self.loss_moving_avg = None
        
    def on_train_begin(self, logs=None):
        # Initialize model optimizer explicitly to avoid 'NoneType' errors
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            log_queue.put("Warning: Optimizer not found in model. Initializing optimizer...")
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_lr),
                loss='mse'
            )
            
            # Run one training step to initialize optimizer
            try:
                dummy_input_shape = list(self.model.input.shape)
                if dummy_input_shape[0] is None:
                    dummy_input_shape[0] = 1
                if dummy_input_shape[1] is None or dummy_input_shape[2] is None:
                    dummy_input_shape[1] = 224
                    dummy_input_shape[2] = 224
                    
                dummy_output_shape = list(self.model.output.shape)
                if dummy_output_shape[0] is None:
                    dummy_output_shape[0] = 1
                if dummy_output_shape[1] is None or dummy_output_shape[2] is None:
                    dummy_output_shape[1] = dummy_input_shape[1] // 4
                    dummy_output_shape[2] = dummy_input_shape[2] // 4
                    
                dummy_x = tf.zeros(dummy_input_shape)
                dummy_y = tf.zeros(dummy_output_shape)
                
                # Run one training step to initialize optimizer
                self.model.train_on_batch(dummy_x, dummy_y)
                log_queue.put("Optimizer initialized in ImprovedAdaptiveLRScheduler")
            except Exception as e:
                log_queue.put(f"Error initializing optimizer in ImprovedAdaptiveLRScheduler: {str(e)}")
        
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
            # Progressive warmup (cubic or similar curve instead of linear)
            # This provides a smoother startup with less initial oscillation
            progress = (epoch + 1) / self.warmup_epochs
            # Cubic warmup curve gives gentler start and faster ramp-up
            warmup_factor = progress**2 * (3 - 2 * progress)  
            lr = self.initial_lr * warmup_factor
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            
            if self.verbose > 0:
                log_queue.put(f"Epoch {epoch+1}: Warmup phase, LR set to {lr:.6f}")
        else:
            # Get current learning rate
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
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
        
        # Update moving average of loss for smoother trend detection
        if self.loss_moving_avg is None:
            self.loss_moving_avg = current_loss
        else:
            self.loss_moving_avg = self.loss_memory_factor * self.loss_moving_avg + \
                                   (1 - self.loss_memory_factor) * current_loss
        
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
            improvement = (self.best_loss - current_loss) / self.best_loss
            self.best_loss = current_loss
            self.wait = 0
            
            # If we've had significant improvement, consider increasing LR
            if epoch > self.warmup_epochs + 5 and improvement > 0.1:
                # Try slightly increasing the learning rate if improvement is substantial
                new_lr = min(current_lr * self.recovery_factor, self.initial_lr)
                if new_lr > current_lr and self.verbose > 0:
                    log_queue.put(f"Significant improvement detected ({improvement:.2%}), slightly increasing LR to {new_lr:.6f}")
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    
                    # Update UI immediately
                    if self.current_lr_var:
                        try:
                            self.current_lr_var.set(f"Current LR: {new_lr:.6f}")
                        except Exception as e:
                            log_queue.put(f"Error updating LR display: {str(e)}")
        else:
            self.wait += 1
            
            # Enhanced trend detection: analyze recent loss history
            # Detect early signs of loss increasing
            if len(self.history['loss']) >= self.trend_detection_window:
                # Calculate trend using the last few epochs
                recent_losses = self.history['loss'][-self.trend_detection_window:]
                
                # Compute the slope of recent losses (positive means increasing)
                if self.trend_detection_window >= 3:
                    # Simple linear regression for trend detection
                    x = list(range(self.trend_detection_window))
                    mean_x = sum(x) / len(x)
                    mean_y = sum(recent_losses) / len(recent_losses)
                    
                    numerator = sum((x[i] - mean_x) * (recent_losses[i] - mean_y) 
                                    for i in range(self.trend_detection_window))
                    denominator = sum((x[i] - mean_x)**2 for i in range(self.trend_detection_window))
                    
                    if denominator != 0:
                        slope = numerator / denominator
                        
                        # Stronger reaction to consistent upward trends
                        if slope > 0:
                            trend_strength = slope / mean_y  # Normalized slope
                            
                            # If trend is clearly upward, reduce LR preemptively
                            if trend_strength > self.early_reaction_threshold:
                                reduction = max(self.reduction_factor, 1.0 - (trend_strength * 2))
                                new_lr = max(current_lr * reduction, self.min_lr)
                                
                                if self.verbose > 0:
                                    log_queue.put(f"Upward loss trend detected (strength: {trend_strength:.4f}), "
                                                f"preemptively reducing LR to {new_lr:.6f}")
                                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                                
                                # Update UI immediately
                                if self.current_lr_var:
                                    try:
                                        self.current_lr_var.set(f"Current LR: {new_lr:.6f}")
                                    except Exception as e:
                                        log_queue.put(f"Error updating LR display: {str(e)}")
                                        
                                self.cooldown_counter = max(1, self.cooldown - 1)
                                self.wait = 0
                                return
            
            # Check for dramatic loss increase
            if len(self.history['loss']) >= 2:
                loss_ratio = current_loss / self.history['loss'][-2]
                
                # If loss increased significantly (>8%), reduce LR more aggressively
                # Lower threshold from 10% to 8% for faster response
                if loss_ratio > 1.08:  
                    # More severe reduction for bigger increases
                    reduction_strength = min(0.9, self.aggressive_reduction * loss_ratio)
                    new_lr = max(current_lr * (1.0 - reduction_strength), self.min_lr)
                    
                    if self.verbose > 0:
                        log_queue.put(f"Significant loss increase detected ({loss_ratio:.2f}x), "
                                     f"aggressively reducing LR to {new_lr:.6f}")
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
                
                # Handle smaller increases (3-8%) - more sensitive than before
                elif loss_ratio > 1.03:
                    # Progressive reduction based on the size of increase
                    reduction_factor = self.reduction_factor + ((loss_ratio - 1.03) / 0.05) * 0.1
                    new_lr = max(current_lr * reduction_factor, self.min_lr)
                    
                    if self.verbose > 0:
                        log_queue.put(f"Small loss increase detected ({loss_ratio:.2f}x), "
                                     f"moderately reducing LR to {new_lr:.6f}")
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    
                    # Update UI immediately
                    if self.current_lr_var:
                        try:
                            self.current_lr_var.set(f"Current LR: {new_lr:.6f}")
                        except Exception as e:
                            log_queue.put(f"Error updating LR display: {str(e)}")
                            
                    self.cooldown_counter = max(1, self.cooldown - 1)
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

# Legacy adaptive learning rate scheduler (kept for backward compatibility)
class AdaptiveLRScheduler(Callback):
    def __init__(self, initial_lr=0.001, min_lr=1e-6, patience=3, 
                 reduction_factor=0.5, warmup_epochs=3, cooldown=2,
                 aggressive_reduction=0.2, recovery_factor=1.1, verbose=1,
                 current_lr_var=None, small_change_threshold=0.05,
                 small_change_factor=0.75):
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
        # New parameters for handling small changes
        self.small_change_threshold = small_change_threshold  # 5% threshold
        self.small_change_factor = small_change_factor  # Less aggressive reduction for small changes
        
    def on_train_begin(self, logs=None):
        # Initialize model optimizer explicitly to avoid 'NoneType' errors
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            log_queue.put("Warning: Optimizer not found in model. Initializing optimizer...")
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_lr),
                loss='mse'
            )
            
            # Run one training step to initialize optimizer
            try:
                dummy_input_shape = list(self.model.input.shape)
                if dummy_input_shape[0] is None:
                    dummy_input_shape[0] = 1
                if dummy_input_shape[1] is None or dummy_input_shape[2] is None:
                    dummy_input_shape[1] = 224
                    dummy_input_shape[2] = 224
                    
                dummy_output_shape = list(self.model.output.shape)
                if dummy_output_shape[0] is None:
                    dummy_output_shape[0] = 1
                if dummy_output_shape[1] is None or dummy_output_shape[2] is None:
                    dummy_output_shape[1] = dummy_input_shape[1] // 2
                    dummy_output_shape[2] = dummy_input_shape[2] // 2
                    
                dummy_x = tf.zeros(dummy_input_shape)
                dummy_y = tf.zeros(dummy_output_shape)
                
                # Run one training step to initialize optimizer
                self.model.train_on_batch(dummy_x, dummy_y)
                log_queue.put("Optimizer initialized in AdaptiveLRScheduler")
            except Exception as e:
                log_queue.put(f"Error initializing optimizer in AdaptiveLRScheduler: {str(e)}")
        
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
            
            # Check for loss change relative to previous epoch
            if len(self.history['loss']) >= 2:
                loss_ratio = current_loss / self.history['loss'][-2]
                
                # If loss increased significantly (>10%), reduce LR more aggressively
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
                
                # Handle smaller increases (5-10%)
                elif loss_ratio > 1.0 + self.small_change_threshold:
                    new_lr = max(current_lr * self.small_change_factor, self.min_lr)
                    if self.verbose > 0:
                        log_queue.put(f"Small loss increase detected ({loss_ratio:.2f}x), moderately reducing LR to {new_lr:.6f}")
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    
                    # Update UI immediately
                    if self.current_lr_var:
                        try:
                            self.current_lr_var.set(f"Current LR: {new_lr:.6f}")
                        except Exception as e:
                            log_queue.put(f"Error updating LR display: {str(e)}")
                            
                    self.cooldown_counter = max(1, self.cooldown - 1)  # Shorter cooldown for small adjustments
                    self.wait = 0
                    return
                
                # Handle plateau (very small change either direction)
                elif abs(loss_ratio - 1.0) < self.small_change_threshold / 2:
                    # Only reduce LR after a few consecutive epochs of small changes
                    if self.wait >= max(2, self.patience - 1):
                        new_lr = max(current_lr * (self.reduction_factor + 0.2), self.min_lr)  # Less aggressive reduction
                        if self.verbose > 0:
                            log_queue.put(f"Loss plateau detected ({loss_ratio:.2f}x), slightly reducing LR to {new_lr:.6f}")
                        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                        
                        # Update UI immediately
                        if self.current_lr_var:
                            try:
                                self.current_lr_var.set(f"Current LR: {new_lr:.6f}")
                            except Exception as e:
                                log_queue.put(f"Error updating LR display: {str(e)}")
                                
                        self.cooldown_counter = max(1, self.cooldown - 1)  # Shorter cooldown
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