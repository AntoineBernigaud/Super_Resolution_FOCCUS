import numpy as np
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import Input, Flatten, LeakyReLU, Reshape
from tensorflow.keras.layers import Concatenate, UpSampling2D, Multiply
from tensorflow.keras.models import Model
import tensorflow as tf

def cgan_disc(img_shape, cond_shape):

    def conv_block(channels, strides=2):
        def block(x):
            x = Conv2D(channels, kernel_size=3, strides=strides, padding="same")(x)
            x = LeakyReLU(0.2)(x)
            return x
        return block

    # Input for the image (real or generated)
    image_in = Input(shape=img_shape, name="sample_in")

    # Input for the conditional image (also of shape img_shape)
    cond_in = Input(shape=cond_shape, name="cond_in")
    
    mask_in = Input(shape=img_shape, name="mask_in")

    masked_image = Multiply()([image_in, mask_in])

    # Processing the input image with convolutional blocks
    x = conv_block(64, strides=1)(masked_image)  # First conv block with stride=1
    x = conv_block(128)(x)                   # Second conv block with default stride=2
    x = conv_block(256)(x)                   # Third conv block with default stride=2
    x = Flatten()(x)                         # Flatten the feature map
    x = Dense(256)(x)                        # Fully connected layer

    # Processing the conditioning image with convolutional blocks
    c = conv_block(64, strides=1)(cond_in)   # First conv block with stride=1
    c = conv_block(128)(c)                   # Second conv block with default stride=2
    c = conv_block(256)(c)                   # Third conv block with default stride=2
    c = Flatten()(c)                         # Flatten the feature map
    c = Dense(256)(c)                        # Fully connected layer

    # Combine the feature maps of the image and the conditional image
    x = Multiply()([x, c])                   # Element-wise multiplication (alternative: Concatenate())

    # Fully connected layers after combination
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)

    # Output of the discriminator: single value (real or fake)
    disc_out = Dense(1, activation="sigmoid")(x)

    # Create the model
    model = Model(inputs=[image_in, cond_in, mask_in], outputs=disc_out)

    return model


def cgan_gen(img_shape, cond_shape, noise_dim=64):

    def up_block(channels):
        def block(x):
            x = UpSampling2D()(x)
            x = Conv2D(channels, kernel_size=3, padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(0.2)(x)
            return x
        return block

    cond_in = Input(shape=cond_shape, name="cond_in")
    noise_in = Input(shape=(noise_dim,), name="noise_in")
    inputs = Concatenate()([Flatten()(cond_in),noise_in])
    
    initial_shape = (img_shape[0]//4, img_shape[1]//4, 256)
    #print("Shape of initial_shape:", tf.shape(initial_shape))

    x = Dense(np.prod(initial_shape))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Reshape(initial_shape)(x)
    x = up_block(256)(x)
    x = up_block(128)(x)
    #print("Shape of x:", tf.shape(x))
    img_out = Conv2D(filters=1, kernel_size=3, padding="same", 
        activation="linear")(x)
    #print("Shape of img_out:", tf.shape(img_out))
    
    return Model(inputs=[cond_in,noise_in], outputs=img_out)
