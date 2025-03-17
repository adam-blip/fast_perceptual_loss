import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Input, MaxPooling2D, BatchNormalization, 
    Add, DepthwiseConv2D, SeparableConv2D, 
    GlobalAveragePooling2D, Concatenate, Multiply,
    Reshape, Layer, Lambda, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2

# Custom layers for serialization
class WeightedAddLayer(tf.keras.layers.Layer):
    """Custom layer for weighted addition with a learnable parameter"""
    def __init__(self, initial_value=0.1, name=None, **kwargs):
        super(WeightedAddLayer, self).__init__(name=name, **kwargs)
        self.initial_value = initial_value
        
    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_value),
            trainable=True
        )
        super(WeightedAddLayer, self).build(input_shape)
        
    def call(self, inputs):
        # inputs should be a list of [x, residual]
        return inputs[0] + inputs[1] * self.gamma
        
    def get_config(self):
        config = super(WeightedAddLayer, self).get_config()
        config.update({'initial_value': self.initial_value})
        return config

class MeanReduceLayer(tf.keras.layers.Layer):
    """Custom layer for reducing mean along specified axes"""
    def __init__(self, axis, keepdims=True, name=None, **kwargs):
        super(MeanReduceLayer, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims
        
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keepdims)
        
    def get_config(self):
        config = super(MeanReduceLayer, self).get_config()
        config.update({
            'axis': self.axis,
            'keepdims': self.keepdims
        })
        return config

class MaxReduceLayer(tf.keras.layers.Layer):
    """Custom layer for reducing max along specified axes"""
    def __init__(self, axis, keepdims=True, name=None, **kwargs):
        super(MaxReduceLayer, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims
        
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=self.axis, keepdims=self.keepdims)
        
    def get_config(self):
        config = super(MaxReduceLayer, self).get_config()
        config.update({
            'axis': self.axis,
            'keepdims': self.keepdims
        })
        return config

# Mish activation function as a Lambda layer function
def mish_activation(x):
    return x * tf.math.tanh(tf.math.softplus(x))

# Enhanced Channel Attention Module
def enhanced_channel_attention(inputs, reduction_ratio=16, name_prefix=''):
    """
    Memory-efficient channel attention module with improved expressiveness
    """
    channels = inputs.shape[-1]
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D(name=f'{name_prefix}_avg_pool')(inputs)
    avg_pool = Reshape((1, 1, channels), name=f'{name_prefix}_reshape_avg')(avg_pool)
    
    # Global max pooling for more expressive attention
    max_pool = MaxReduceLayer(
        axis=[1, 2], 
        keepdims=True, 
        name=f'{name_prefix}_max_pool'
    )(inputs)
    
    # Combine pooled features
    pooled = Add(name=f'{name_prefix}_pool_combine')([avg_pool, max_pool])
    
    # Bottleneck MLP with minimal parameters
    reduced_channels = max(1, channels // reduction_ratio)
    fc1 = Conv2D(reduced_channels, kernel_size=1, use_bias=False, 
                kernel_initializer=HeNormal(seed=42),
                name=f'{name_prefix}_fc1')(pooled)
    relu = Activation('relu', name=f'{name_prefix}_relu')(fc1)
    fc2 = Conv2D(channels, kernel_size=1, use_bias=False,
                kernel_initializer=HeNormal(seed=42),
                name=f'{name_prefix}_fc2')(relu)
    
    # Apply sigmoid activation
    attention = Activation('sigmoid', name=f'{name_prefix}_sigmoid')(fc2)
    
    # Apply attention weights to input feature map
    return Multiply(name=f'{name_prefix}_multiply')([inputs, attention])

# Ghost Module - efficient feature extraction with fewer parameters
def ghost_module(inputs, output_channels, name_prefix=''):
    """
    Ghost Module: Uses fewer parameters by generating more features through
    cheap operations rather than expensive convolutions.
    """
    # Determine primary channels - typically 1/2 of output channels
    ratio = 2
    primary_channels = output_channels // ratio
    
    # Primary convolution with fewer filters
    primary = Conv2D(primary_channels, 1, padding='same',
                   kernel_initializer=HeNormal(seed=42),
                   use_bias=False,
                   name=f'{name_prefix}_ghost_primary')(inputs)
    primary = BatchNormalization(momentum=0.9, name=f'{name_prefix}_ghost_bn1')(primary)
    primary = Lambda(mish_activation, name=f'{name_prefix}_ghost_mish1')(primary)
    
    # Cheap operations to generate more features
    cheap_ops = []
    cheap_ops.append(primary)  # Include the primary features
    
    # Generate remaining features through depthwise convolutions
    for i in range(ratio - 1):
        cheap = DepthwiseConv2D(3, padding='same',
                              depthwise_initializer=HeNormal(seed=42),
                              use_bias=False,
                              name=f'{name_prefix}_ghost_dw{i+1}')(primary)
        cheap = BatchNormalization(momentum=0.9, name=f'{name_prefix}_ghost_bn{i+2}')(cheap)
        cheap = Lambda(mish_activation, name=f'{name_prefix}_ghost_mish{i+2}')(cheap)
        cheap_ops.append(cheap)
    
    # Concatenate all features
    ghost_features = Concatenate(name=f'{name_prefix}_ghost_concat')(cheap_ops)
    
    return ghost_features

# Spatial Attention Module
def spatial_attention(inputs, kernel_size=7, name_prefix=''):
    """
    Efficient spatial attention module to highlight important regions
    """
    # Average pooling along channel dimension
    avg_pool = MeanReduceLayer(
        axis=-1, 
        keepdims=True,
        name=f'{name_prefix}_spatial_avg'
    )(inputs)
    
    # Max pooling along channel dimension
    max_pool = MaxReduceLayer(
        axis=-1, 
        keepdims=True,
        name=f'{name_prefix}_spatial_max'
    )(inputs)
    
    # Concatenate pooled features
    concat = Concatenate(name=f'{name_prefix}_spatial_concat')([avg_pool, max_pool])
    
    # Convolutional layer to generate spatial attention map
    spatial_map = Conv2D(1, kernel_size, padding='same', 
                       kernel_initializer=HeNormal(seed=42),
                       use_bias=False,
                       name=f'{name_prefix}_spatial_conv')(concat)
    
    # Apply sigmoid activation
    spatial_map = Activation('sigmoid', name=f'{name_prefix}_spatial_sigmoid')(spatial_map)
    
    # Apply spatial attention
    return Multiply(name=f'{name_prefix}_spatial_multiply')([inputs, spatial_map])

# Selective Kernel Unit - adaptive receptive field
def selective_kernel_unit(inputs, channels, name_prefix=''):
    """
    Selective Kernel Unit - adaptively adjusts receptive field by fusing 
    multiple kernels with different sizes.
    """
    # First branch - standard 3x3 conv
    branch1 = SeparableConv2D(channels, 3, padding='same',
                            depthwise_initializer=HeNormal(seed=42),
                            pointwise_initializer=HeNormal(seed=42),
                            name=f'{name_prefix}_sk_3x3')(inputs)
    branch1 = BatchNormalization(momentum=0.9, name=f'{name_prefix}_sk_bn1')(branch1)
    branch1 = Lambda(mish_activation, name=f'{name_prefix}_sk_mish1')(branch1)
    
    # Second branch - dilated 3x3 conv (effective 5x5 receptive field)
    branch2 = SeparableConv2D(channels, 3, padding='same', dilation_rate=(2, 2),
                            depthwise_initializer=HeNormal(seed=42),
                            pointwise_initializer=HeNormal(seed=42),
                            name=f'{name_prefix}_sk_dil')(inputs)
    branch2 = BatchNormalization(momentum=0.9, name=f'{name_prefix}_sk_bn2')(branch2)
    branch2 = Lambda(mish_activation, name=f'{name_prefix}_sk_mish2')(branch2)
    
    # Fuse branches with simple addition instead of attention for simplicity
    output = Add(name=f'{name_prefix}_sk_combine')([branch1, branch2])
    
    return output

# Create the enhanced FastPerceptualLoss model with integrated improvements
def create_fast_perceptual_model(input_shape=(None, None, 3)):
    """
    Enhanced model with advanced architectural components:
    - Ghost Modules for parameter efficiency
    - Improved gradient flow with mish activations and skip connections
    - Output dimensions matching VGG19 block3_conv3 (1/4 of input size)
    """
    # Advanced initialization strategy
    kernel_init = HeNormal(seed=42)
    kernel_reg = l2(1e-5)
    
    # Input layer
    inputs = Input(shape=input_shape, name="input_image")
    
    # First block - Ghost module for efficiency
    x = ghost_module(inputs, 16, name_prefix='block1')
    
    # Skip connection for residual learning
    skip1 = x
    
    # FIRST MAX POOLING - reduces dimensions by 2x
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    
    # Second block - Selective Kernel Unit for adaptive features
    x = selective_kernel_unit(x, 32, name_prefix='block2')
    
    # Skip connection
    residual2 = x
    
    # Add spatial attention
    x = spatial_attention(x, kernel_size=5, name_prefix='sa1')
    
    # Residual connection
    x = Add(name="add1")([x, residual2])
    
    # SECOND MAX POOLING - total reduction now 4x (matches VGG19)
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    
    # Final feature refinement block 
    x = SeparableConv2D(64, 3, padding='same',
                      depthwise_initializer=kernel_init,
                      pointwise_initializer=kernel_init,
                      name="sep_conv3_1")(x)
    x = BatchNormalization(momentum=0.9, name="bn3_1")(x)
    x = Lambda(mish_activation, name="mish3_1")(x)
    
    # Multi-level feature fusion
    # Take first skip connection, downsample with fixed size
    skip1_down = MaxPooling2D(pool_size=(4, 4), name="skip1_down")(skip1)
    skip1_proj = Conv2D(64, 1, padding='same',
                      kernel_initializer=kernel_init,
                      name="skip1_proj")(skip1_down)
    
    # Combine with main path
    x = Add(name="multi_level_fusion")([x, skip1_proj])
    
    # Final enhanced channel attention
    x = enhanced_channel_attention(x, reduction_ratio=8, name_prefix='final_ca')
    
    # Final 1x1 projection to match VGG19 features dimension (256)
    x = Conv2D(256, 1, padding='same', 
             kernel_initializer=kernel_init,
             kernel_regularizer=kernel_reg,
             name="final_proj")(x)
    x = BatchNormalization(momentum=0.9, name="final_bn")(x)
    x = Lambda(mish_activation, name="final_mish")(x)
    
    # The final output matches VGG19 block3_conv3 output (256 filters, 1/4 spatial dim)
    model = Model(inputs, x, name="EnhancedFastPerceptualLoss")
    
    # Print model summary and parameter count
    base_params = 48000  # Approximate parameter count of original model
    current_params = model.count_params()
    overhead = (current_params - base_params) / base_params * 100
    
    print(f"Enhanced model created with {current_params:,} parameters")
    print(f"Parameter overhead: {overhead:.2f}% compared to original model")
    print(f"Input shape: {model.input.shape}, Output shape: {model.output.shape}")
    
    return model