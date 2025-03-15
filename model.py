import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Input, MaxPooling2D, BatchNormalization, ReLU, 
    Add, DepthwiseConv2D, SeparableConv2D, Dropout, 
    GlobalAveragePooling2D, Dense, Concatenate, Multiply,
    Reshape, Layer, Lambda, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2

# Lightweight Channel Attention Module
def lightweight_channel_attention(inputs, reduction_ratio=16, name_prefix=''):
    """
    Memory-efficient channel attention module with minimal parameters
    
    Args:
        inputs: Input feature map
        reduction_ratio: Reduction ratio for the MLP
        name_prefix: Prefix for layer names
        
    Returns:
        Channel-refined feature map
    """
    channels = inputs.shape[-1]
    
    # Global average pooling (uses no parameters)
    avg_pool = GlobalAveragePooling2D(name=f'{name_prefix}_avg_pool')(inputs)
    avg_pool = Reshape((1, 1, channels), name=f'{name_prefix}_reshape')(avg_pool)
    
    # Small MLP with extreme bottleneck for parameter efficiency
    # Reduce channels even more for extreme parameter reduction
    reduced_channels = max(1, channels // reduction_ratio)
    
    # First dense layer with significant reduction
    fc1 = Conv2D(reduced_channels, kernel_size=1, use_bias=False, 
                kernel_initializer=HeNormal(seed=42),
                name=f'{name_prefix}_fc1')(avg_pool)
    relu = ReLU(name=f'{name_prefix}_relu')(fc1)
    
    # Second dense layer to restore original channels
    fc2 = Conv2D(channels, kernel_size=1, use_bias=False,
                kernel_initializer=HeNormal(seed=42),
                name=f'{name_prefix}_fc2')(relu)
    
    # Apply sigmoid activation
    attention = Activation('sigmoid', name=f'{name_prefix}_sigmoid')(fc2)
    
    # Apply attention weights to input feature map
    return Multiply(name=f'{name_prefix}_multiply')([inputs, attention])

# Create the enhanced FastPerceptualLoss model with improved architecture but reduced parameters
def create_fast_perceptual_model(input_shape=(None, None, 3)):
    """
    Creates a memory-efficient model using modern techniques but with greatly reduced parameters:
    - Minimal channel attention for better feature emphasis
    - Depthwise separable convolutions for parameter efficiency
    - Reduced filter counts across all layers
    - Strategic use of residual connections
    - Batch normalization for training stability
    """
    # Advanced initialization strategy
    kernel_init = HeNormal(seed=42)  # He initialization for ReLU activations
    kernel_reg = l2(1e-5)  # L2 regularization to prevent overfitting
    
    # Input layer
    inputs = Input(shape=input_shape, name="input_image")
    
    # First block with standard convolution but reduced filters
    x = Conv2D(8, 3, padding='same', 
              kernel_initializer=kernel_init, 
              kernel_regularizer=kernel_reg,
              name="conv1_1")(inputs)
    x = BatchNormalization(momentum=0.9, name="bn1_1")(x)
    x = ReLU(name="relu1_1")(x)
    
    # Store for residual connection
    residual1 = x
    
    # Block 2: Efficient depthwise separable convolution block
    # Depthwise convolution (spatial features)
    x = DepthwiseConv2D(3, padding='same', 
                       depthwise_initializer=kernel_init,
                       depthwise_regularizer=kernel_reg,
                       name="dw_conv2_1")(x)
    x = BatchNormalization(momentum=0.9, name="bn2_1")(x)
    x = ReLU(name="relu2_1")(x)
    
    # Pointwise convolution (channel features) with reduced filters
    x = Conv2D(16, 1, padding='same', 
              kernel_initializer=kernel_init,
              kernel_regularizer=kernel_reg,
              name="pw_conv2_1")(x)
    x = BatchNormalization(momentum=0.9, name="bn2_2")(x)
    x = ReLU(name="relu2_2")(x)
    
    # Apply lightweight channel attention (with higher reduction ratio to save parameters)
    x = lightweight_channel_attention(x, reduction_ratio=16, name_prefix='ca1')
    
    # Add residual connection with projection (due to different filter sizes)
    projection1 = Conv2D(16, 1, padding='same', 
                        kernel_initializer=kernel_init,
                        kernel_regularizer=kernel_reg,
                        name="proj1")(residual1)
    x = Add(name="add1")([x, projection1])
    
    # Downsampling with Max Pooling
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    
    # Block 3: Deeper features with depthwise separable convolutions
    # Depthwise convolution
    x = DepthwiseConv2D(3, padding='same', 
                       depthwise_initializer=kernel_init,
                       depthwise_regularizer=kernel_reg,
                       name="dw_conv3_1")(x)
    x = BatchNormalization(momentum=0.9, name="bn3_1")(x)
    x = ReLU(name="relu3_1")(x)
    
    # Pointwise convolution with reduced filters
    x = Conv2D(32, 1, padding='same', 
              kernel_initializer=kernel_init,
              kernel_regularizer=kernel_reg,
              name="pw_conv3_1")(x)
    x = BatchNormalization(momentum=0.9, name="bn3_2")(x)
    x = ReLU(name="relu3_2")(x)
    
    # Store for residual connection
    residual2 = x
    
    # Second set of depthwise separable convolutions
    x = SeparableConv2D(32, 3, padding='same',
                       depthwise_initializer=kernel_init,
                       pointwise_initializer=kernel_init,
                       name="sep_conv3_2")(x)
    x = BatchNormalization(momentum=0.9, name="bn3_3")(x)
    x = ReLU(name="relu3_3")(x)
    
    # Apply lightweight channel attention
    x = lightweight_channel_attention(x, reduction_ratio=16, name_prefix='ca2')
    
    # Residual connection
    x = Add(name="add2")([x, residual2])
    
    # Downsampling with Max Pooling
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    
    # Final convolution block with reduced filters and enlarged receptive field
    x = Conv2D(64, 3, padding='same', 
              dilation_rate=2,
              kernel_initializer=kernel_init, 
              kernel_regularizer=kernel_reg,
              name="conv4_1")(x)
    x = BatchNormalization(momentum=0.9, name="bn4_1")(x)
    x = ReLU(name="relu4_1")(x)
    
    # Final 1x1 projection to match VGG19 features dimension (256)
    x = Conv2D(256, 1, padding='same', 
              kernel_initializer=kernel_init,
              kernel_regularizer=kernel_reg,
              name="conv5_1")(x)
    x = BatchNormalization(momentum=0.9, name="bn5_1")(x)
    x = ReLU(name="relu5_1")(x)
    
    # The final output matches VGG19 block3_conv3 output (256 filters)
    return Model(inputs, x, name="FastPerceptualLoss")