from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import GlobalAveragePooling2D,MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense, Layer
from keras.layers import concatenate, Input, dot, Lambda, Maximum
from keras import backend as K
from keras.applications import VGG16
import tensorflow as tf
import numpy as np
import cv2
import imutils

def vgg16_model(trainable=True):
    base_model = VGG16(False, "imagenet")
    train_from_layer = -2
    for layer in base_model.layers[:train_from_layer]:
        layer.trainable = False
        print("{} is not trainable".format(layer.name))
    for layer in base_model.layers[train_from_layer:]:
        #layer.trainable = True
        layer.trainable = False
        print("{} is trainable".format(layer.name))
    last_conv_layer = base_model.get_layer("block5_conv3")
    x = GlobalAveragePooling2D()(last_conv_layer.output)
    #x = Flatten()(last_conv_layer.output)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)        
    predictions = Dense(1, activation="sigmoid")(x)
    return Model(base_model.input, predictions)

def small_vgg(width, height, depth, classes, params=[32,64,128,1024]):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(params[0], (3, 3), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(params[1], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(params[1], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(params[2], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(params[2], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(params[3]))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

def vgg(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

def vgg_fourier_mid(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    inputShape = (height, width, depth)
    chanDim = -1

    input1 = Input(inputShape)
    input2 = Input((width, height, 1))
   
    c1_1 = Conv2D(32, (3, 3), padding="same", activation='relu')(input1)
    bn1_1 = BatchNormalization(axis=chanDim)(c1_1)
    mp1_1 = MaxPooling2D(pool_size=(3, 3))(bn1_1)
    do1_1 = Dropout(0.25)(mp1_1)
    c1_2 = Conv2D(32, (3, 3), padding="same", activation='relu')(input2)
    bn1_2 = BatchNormalization(axis=chanDim)(c1_2)
    mp1_2 = MaxPooling2D(pool_size=(3, 3))(bn1_2)
    do1_2 = Dropout(0.25)(mp1_2)
            
            
    c2_1 = Conv2D(64, (3, 3), padding="same", activation='relu')(do1_1)
    bn2_1 = BatchNormalization(axis=chanDim)(c2_1)
    c2__1 = Conv2D(64, (3, 3), padding="same", activation='relu')(bn2_1)
    bn2__1 = BatchNormalization(axis=chanDim)(c2__1)
    mp2_1 = MaxPooling2D(pool_size=(2, 2))(bn2__1)
    do2_1 = Dropout(0.25)(mp2_1)
    c2_2 = Conv2D(64, (3, 3), padding="same", activation='relu')(do1_2)
    bn2_2 = BatchNormalization(axis=chanDim)(c2_2)
    c2__2 = Conv2D(64, (3, 3), padding="same", activation='relu')(bn2_2)
    bn2__2 = BatchNormalization(axis=chanDim)(c2__2)
    mp2_2 = MaxPooling2D(pool_size=(2, 2))(bn2__2)
    do2_2 = Dropout(0.25)(mp2_2)
          
    merge = concatenate([do2_1,do2_2])

    c3 = Conv2D(64, (3, 3), padding="same", activation='relu')(merge)
    c3 = BatchNormalization(axis=chanDim)(c3)
    c3 = Conv2D(64, (3, 3), padding="same", activation='relu')(c3)
    c3 = BatchNormalization(axis=chanDim)(c3)
    c3 = MaxPooling2D(pool_size=(2, 2))(c3)
    c3 = Dropout(0.25)(c3)

    c4 = Conv2D(128, (3, 3), padding="same", activation='relu')(c3)
    c4 = BatchNormalization(axis=chanDim)(c4)
    c4 = Conv2D(128, (3, 3), padding="same", activation='relu')(c4)
    c4 = BatchNormalization(axis=chanDim)(c4)
    c4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c4 = Dropout(0.25)(c4)

    c5 = Conv2D(256, (3, 3), padding="same", activation='relu')(c4)
    c5 = BatchNormalization(axis=chanDim)(c5)
    c5 = Conv2D(256, (3, 3), padding="same", activation='relu')(c5)
    c5 = BatchNormalization(axis=chanDim)(c5)
    c5 = MaxPooling2D(pool_size=(2, 2))(c5)
    c5 = Dropout(0.25)(c5)

    c6 = Conv2D(512, (3, 3), padding="same", activation='relu')(c5)
    c6 = BatchNormalization(axis=chanDim)(c6)
    c6 = Conv2D(512, (3, 3), padding="same", activation='relu')(c6)
    c6 = BatchNormalization(axis=chanDim)(c6)
    c6 = MaxPooling2D(pool_size=(2, 2))(c6)
    c6 = Dropout(0.25)(c6)

    out = Flatten()(c6)
    out = Dense(1024, activation='relu')(out)
    out = BatchNormalization()(out)
    out =  Dropout(0.5)(out)


    out = Dense(classes, activation='softmax')(out)

    model = Model(inputs=[input1,input2], outputs=out)
    # return the constructed network architecture
    return model


def vgg_fourier_end(width, height, depth, classes, ft_shape=(75,75)):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    inputShape = (height, width, depth)
    chanDim = -1

    input1 = Input(inputShape)
    input2 = Input((*ft_shape, 1))

    c1_1 = Conv2D(32, (3, 3), padding="same", activation='relu')(input1)
    bn1_1 = BatchNormalization(axis=chanDim)(c1_1)
    mp1_1 = MaxPooling2D(pool_size=(3, 3))(bn1_1)
    do1_1 = Dropout(0.25)(mp1_1)

    c1_2 = Conv2D(1, ft_shape, padding="same", activation='relu')(input2)
    do1_2 = Dropout(0.3)(c1_2)
    fl_2 = Flatten()(do1_2)
    dn_2 = Dense(1024, activation='relu')(fl_2)


    c2_1 = Conv2D(64, (3, 3), padding="same", activation='relu')(do1_1)
    bn2_1 = BatchNormalization(axis=chanDim)(c2_1)
    c2__1 = Conv2D(64, (3, 3), padding="same", activation='relu')(bn2_1)
    bn2__1 = BatchNormalization(axis=chanDim)(c2__1)
    mp2_1 = MaxPooling2D(pool_size=(2, 2))(bn2__1)
    do2_1 = Dropout(0.25)(mp2_1)


    c3 = Conv2D(64, (3, 3), padding="same", activation='relu')(do2_1)
    c3 = BatchNormalization(axis=chanDim)(c3)
    c3 = Conv2D(64, (3, 3), padding="same", activation='relu')(c3)
    c3 = BatchNormalization(axis=chanDim)(c3)
    c3 = MaxPooling2D(pool_size=(2, 2))(c3)
    c3 = Dropout(0.25)(c3)

    c4 = Conv2D(128, (3, 3), padding="same", activation='relu')(c3)
    c4 = BatchNormalization(axis=chanDim)(c4)
    c4 = Conv2D(128, (3, 3), padding="same", activation='relu')(c4)
    c4 = BatchNormalization(axis=chanDim)(c4)
    c4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c4 = Dropout(0.25)(c4)

    c5 = Conv2D(256, (3, 3), padding="same", activation='relu')(c4)
    c5 = BatchNormalization(axis=chanDim)(c5)
    c5 = Conv2D(256, (3, 3), padding="same", activation='relu')(c5)
    c5 = BatchNormalization(axis=chanDim)(c5)
    c5 = MaxPooling2D(pool_size=(2, 2))(c5)
    c5 = Dropout(0.25)(c5)

    c6 = Conv2D(512, (3, 3), padding="same", activation='relu')(c5)
    c6 = BatchNormalization(axis=chanDim)(c6)
    c6 = Conv2D(512, (3, 3), padding="same", activation='relu')(c6)
    c6 = BatchNormalization(axis=chanDim)(c6)
    c6 = MaxPooling2D(pool_size=(2, 2))(c6)
    c6 = Dropout(0.25)(c6)

    fl_1 = Flatten()(c6)
    dn_1 = Dense(1024, activation='relu')(fl_1)

    merge = concatenate([dn_1,dn_2])
    out = Dense(1024, activation='relu')(merge)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)

    out = Dense(classes, activation='softmax')(out)

    model = Model(inputs=[input1, input2], outputs=out)
    # return the constructed network architecture
    return model

def vgg2(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1


    model.add(Conv2D(64, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model


def large_vgg(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

def add_top(model, classes):
    chanDim = -1

    model.add(Conv2D(1024, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(1024, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model
def add_fs_top(model, classes, size=8192):
    model.add(Dense(size))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

def add_top_small(model, classes):
    chanDim = -1

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

class FFT_Filter(Layer):

    def __init__(self, **kwargs):
        super(FFT_Filter, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True
                                      )
        super(FFT_Filter, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        
        ftimage = np.fft.fft2(x.eval())
        ftimage = np.fft.fftshift(ftimage)
        fft = tf.cast(ftimage,dtype=tf.complex64)
        real = tf.real(fft)
        imag = tf.imag(fft)
        fil = real*self.kernel
        full = tf.complex(fil, imag)
        return abs(tf.ifft2d(full))

    def compute_output_shape(self, input_shape):
        return input_shape

class FFT_IN(Layer):

    def __init__(self, **kwargs):
        super(FFT_IN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,) + input_shape[1:],
                                      initializer='he_normal',
                                      trainable=True
                                      )
        super(FFT_IN, self).build(input_shape)  # Be sure to call this somewhere!

  
    def call(self, x):
        fft = tf.fft2d(tf.cast(x, dtype=tf.complex64))
        real =tf.real(fft)*tf.sigmoid(self.kernel) #because you want to cut frequencies, not enhance them
        imag = tf.imag(fft)
        full = tf.complex(real,imag)
        full = abs(tf.ifft2d(full))
        self.kernel = tf.nn.l2_normalize(self.kernel)
        return full

    def compute_output_shape(self, input_shape):
        return input_shape

    

from keras.layers.core import Layer
import tensorflow as tf

class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 localization_net,
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))

    def call(self, X, mask=None):
        affine_transformation = self.locnet.call(X)
        output = self._transform(affine_transformation, X, self.output_size)
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x = .5*(x + 1.0)*(width_float)
        y = .5*(y + 1.0)*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, affine_transformation, input_shape, output_size):
        batch_size = tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[1]
        width = tf.shape(input_shape)[2]
        num_channels = tf.shape(input_shape)[3]

        affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))

        affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
        affine_transformation = tf.cast(affine_transformation, 'float32')

        width = tf.cast(width, dtype='float32')
        height = tf.cast(height, dtype='float32')
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1]) # flatten?

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))

        transformed_grid = tf.matmul(affine_transformation, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                                x_s_flatten,
                                                y_s_flatten,
                                                output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                output_height,
                                                                output_width,
                                                                num_channels))
        return transformed_image
    
    
class RotTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 localization_net,
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(RotTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))

    def call(self, X, mask=None):
        affine_transformation = self.locnet.call(X)#only one value that is the angle
        output = self._transform(affine_transformation, X, self.output_size)
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x = .5*(x + 1.0)*(width_float)
        y = .5*(y + 1.0)*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    #just one value
    def _transform(self, affine_transformation, input_shape, output_size):
        batch_size = tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[1]
        width = tf.shape(input_shape)[2]
        num_channels = tf.shape(input_shape)[3]
        
        
        
        affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2))
        
        sum_affine = tf.atan2(affine_transformation[:,0],affine_transformation[:,1])
        #tf.reduce_sum(affine_transformation, reduction_indices= [1])*np.pi
        
        
        afs_transformation= tf.stack([tf.cos(sum_affine),-tf.sin(sum_affine),sum_affine*0,  
                               tf.sin(sum_affine),tf.cos(sum_affine),sum_affine*0])
        
        #afs_transformation = affine_transformation
        
        #tf.stack([tf.cos(affine_transformation),tf.sin(affine_transformation),
        #          tf.sin(affine_transformation),tf.sin(affine_transformation)]
        
        afs_transformation = tf.reshape(afs_transformation, shape=(batch_size,2,3))

        afs_transformation = tf.reshape(afs_transformation, (-1, 2, 3))
        afs_transformation = tf.cast(afs_transformation, 'float32')
        
        #affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))

        #affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
        #affine_transformation = tf.cast(affine_transformation, 'float32')

        
        
       

        width = tf.cast(width, dtype='float32')
        height = tf.cast(height, dtype='float32')
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1]) # flatten?

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))

        transformed_grid = tf.matmul(afs_transformation, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                                x_s_flatten,
                                                y_s_flatten,
                                                output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                output_height,
                                                                output_width,
                                                                num_channels))
        return transformed_image
        
    

#another rotation approach    
    




class Convolution2D_8(Conv2D):
    
    def __init__(self,filters,kernel_size,positions=16,**kwargs):
        self.filters= filters
        self.kernel_size=kernel_size
        self.positions = positions
        
        self.degrees = list(range(0,360,int(360/positions)))
         
        super(Convolution2D_8, self).__init__(filters,kernel_size,**kwargs)
    
    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])


    def _interpolate(self, image, x, y, output_size):
        batch_size = 1#tf.shape(image)[0]
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        num_channels = tf.shape(image)[2]

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x = .5*(x + 1.0)*(width_float)
        y = .5*(y + 1.0)*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    #just one value
    def _transform(self, rads, input_shape, output_size):
        batch_size = 1#tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[0]
        width = tf.shape(input_shape)[1]
        num_channels = tf.shape(input_shape)[2]
              

        afs_transformation= tf.stack([tf.cos(rads),-tf.sin(rads),tf.sin(rads)*0,  
                               tf.sin(rads),tf.cos(rads),tf.sin(rads)*0])
        
        afs_transformation = tf.reshape(afs_transformation, shape=(batch_size,2,3))

        afs_transformation = tf.reshape(afs_transformation, (-1, 2, 3))
        afs_transformation = tf.cast(afs_transformation, 'float32')
        

        width = tf.cast(width, dtype='float32')
        height = tf.cast(height, dtype='float32')
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1]) # flatten?

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))

        transformed_grid = tf.matmul(afs_transformation, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                                x_s_flatten,
                                                y_s_flatten,
                                                output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                output_height,
                                                                output_width,
                                                                num_channels))
        return transformed_image
    
     #rotate each of the filters and take those with higher output.
    def shift_rotate(self, w, radiants):
        
        #nb_filters = tf.shape(w)[3]
        
        shape = w.get_shape()
        
        out_w = []
        for i in range(self.filters):
            
            intermediate_filter= self._transform(radiants,w[:,:,:,i],tf.shape(w[:,:,:,i]))
            out_w.append(intermediate_filter)            
         
        

        return tf.stack(out_w,axis=4)
    
    
    #brute force
    def call(self, x, mask=None):
        
        w = self.kernel 
        w_rot = []
        #print("here")    
        for i in self.degrees:
            w_inter = self.shift_rotate(w, np.deg2rad(i))
            #print(w_inter)
            w_rot.append(tf.squeeze(w_inter))
            #print("here3")

        
        list_output = []
        list_norm = []
        
        for w_i in w_rot:
           
            output=tf.keras.backend.conv2d(x, w_i, strides=self.strides,padding=self.padding)
            list_output.append(output)
            #list_norm.append(tf.norm(output))
        
        outputs = tf.stack(list_output)
        #outputs_norm = tf.stack(list_norm)
        #idx = tf.argmax(outputs_norm)

        #print("here4")
        #output =tf.nn.max_pool3d(outputs,ksize=[8,1,1,1,8],strides=[8,1,1,1,8],padding="SAME")# .max(outputs, 0)
        #output = outputs[idx] #K.max(outputs,0)
        output = tf.reduce_max(outputs, axis=0)
        
        #print(output)
        if self.use_bias:
            output += tf.reshape(self.bias , (1, 1, 1, self.filters))
                
        output = self.activation(output)
        
        return output
    
                                  
                             
    
 
 
    
    

from tensorflow.python.framework import ops
import numpy as np

class Rot2D(Layer):
    
    
    def __init__(self,**kwargs):
        super(Rot2D,self).__init__(**kwargs)
        #self.internal_shape = internal_shape
    
    def build(self,input_shape):
        self.W = self.add_weight(input_shape[1:],name='kernel',
                                      
                                      initializer='uniform',
                                      trainable=True
                                      )
        self.internal_shape=input_shape[1:]
        
        super(Rot2D, self).build(input_shape)  # Be sure to call this somewhere!

    def my_rot(self,W,x):
        import sys
        list_angle = []
        
        linear_comb = np.dot(x[:,:,:,0],W[:,:,0]) # must have at least one.
        
        for i in range(1,np.shape(x)[3]):
            linear_comb = linear_comb+ np.dot(x[:,:,:,i],W[:,:,i])
            #linear_comb = linear_comb+ np.dot(x[:,:,:,2],W[:,:,2])
        #print("here")
        #print(tf.shape(linear_comb))
        
        #angle =[np.arccos(np.sum(el)/np.linalg.norm(el)) for el in inter] 
        
        #print("angle"+str(angle), file=sys.stderr)
        angle = np.sum(linear_comb, axis=(1,2))
        
        #print("angle"+str(angle), file=sys.stderr)
        
        #rotations = np.zeros(len(angle))
        
        #for i in range(0,len(angle)):
        #    val = np.arccos(angle[i])
        #    if not np.isnan(val):
         #       rotations[i] = val
        
        #rotations = [np.arccos(i) for i in angle]
        #print("rot"+str(rotations), file=sys.stderr)
        
        for i in range(0,len(x)):
            
            #rotation_rad = angle[i]
            #if np.isnan(angle[i]):
            #    rotation_rad = 0
                
            myangle = int(np.rad2deg(angle[i])%360)
            #print("angle"+str(myangle), file=sys.stderr)
            rotated = imutils.rotate(np.squeeze(x[i]),myangle )
            list_angle.append(rotated)
        
        return W,np.array(list_angle)     
    
    
    def _MySquareGrad(self,op, grad,grad1):
        W = op.inputs[0]
        x = op.inputs[1]
        
        print("OP, W, x")
        print(tf.shape(W))
        print(tf.shape(x))
        print("GRAD")
        print(tf.shape(grad))
        #jacobian = ComputeJacobian(x)  # compute jacobian in x
        #mygrad = y[:,:,0]*grad[:,:,:]
        #print(mygrad)
        #mygrad2 = mygrad[:,:,1]*grad[:,:,1]
        
        #mygrad3 = mygrad[:,:,2]*grad[:,:,2]
        #mystuff = tf.stack([mygrad,mygrad2,mygrad3], axis=2)
        #
        
        gradient_exp = -x*grad*1/(1-tf.sqrt(tf.square(x*W)))
        var_ret =tf.reduce_sum( gradient_exp,reduction_indices=[0])
        
        return var_ret,grad1*0#x*tf.reduce_sum(grad,reduction_indices=[0]), grad*0# # np.dot(jacobian.T, grad)
    
    def py_func(self,func, inp, Tout, stateful=True, name=None, grad=None):
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
        tf.RegisterGradient(rnd_name)(self._MySquareGrad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    
 
    def call(self, x):
        #three channels
        
        #rot = tf.stack([(tf.cos(angle), -tf.sin(angle)), (tf.sin(angle), tf.cos(angle))], axis=0)
        #print(type(rot))
        #rot = tf.cast(rot, tf.float32)
        #pred = tf.nn.sigmoid(self.W) # Softmax to regress a rotation
        #pred2 = tf.nn.sigmoid(angle)
        _,res= self.py_func(self.my_rot,[self.W,x],[tf.float32, tf.float32])

        myres=tf.reshape(res,(tf.shape(res)[0],self.internal_shape[0],self.internal_shape[1],self.internal_shape[2]))
        #myz= tf.zeros((128,75,75,3))
        #myz = myz+res
        
        #print(tf.shape(res))
        #tf.Session().run(self.W)
        #tf.Session().run(angle)
        #myr=tf.Session().run(rot)
        #res = tf.matmul(x,rot)
        #res= tf.keras.preprocessing.image.apply_transform( x, rot , channel_axis=0, fill_mode='nearest', cval=0.0)
        return myres#*pred/pred
    
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape)

class FFT_OUT(Layer):

    def __init__(self, **kwargs):
        super(FFT_OUT, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True
                                      )
        super(FFT_OUT, self).build(input_shape)  # Be sure to call this somewhere!

    def my_ifft(self,x):
        myx= np.fft.ifft2(x)
        return myx
    
        
        
    def call(self, x):
        real = tf.real(x)*tf.sigmoid(self.kernel)
        imag = tf.imag(x)
        #fil = real
        full = tf.complex(real, imag)
        #exit = tf.py_func(self.my_ifft,[full],tf.complex64)
        myvar = abs(tf.ifft2d(full))
        #myvar=exit
        return myvar #tf.cast(exit,dtype=tf.float32)*self.kernel
        #return #exit 

    def compute_output_shape(self, input_shape):
        return input_shape


    
    
    
def super_small_conv(classes, shape=(75,75), params=[16,32,64,256], kernel_size=(3,3)):
    chanDim = -1

    model = Sequential()
    model.add(Conv2D(filters=params[0], kernel_size=kernel_size, 
                              padding="same",input_shape=(*shape,3), name="Conv8", activation="relu"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    #model.add(Dropout(0.25))
    #model.add(FFT_OUT(name='FFT_OUT'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(classes,activation='softmax'))

    return model

 

def super_small_rot(classes, shape=(75,75), params=[16,32,64,256], kernel_size=(3,3),positions=16):
    chanDim = -1

    model = Sequential()
    #model.add(Rot2D(name="ROT",input_shape=(*shape,3)))
    #model.add(Conv2D(filters=params[0], kernel_size=kernel_size, padding="same", name="Conv8", activation="relu"))
    model.add(Convolution2D_8(filters=params[0], kernel_size=kernel_size, positions=positions,
                              padding="same",input_shape=(*shape,3), name="Conv8", activation="relu"))
    
    model.add(Convolution2D_8(filters=params[0], kernel_size=kernel_size, positions=positions,
                              padding="same", name="Conv8_2", activation="relu"))

    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    
    model.add(Convolution2D_8(filters=params[1], kernel_size=(4,4), positions=positions,
                              padding="same", name="Conv8_3", activation="relu"))
    
    model.add(Convolution2D_8(filters=params[1], kernel_size=(4,4), positions=positions,
                              padding="same", name="Conv8_4", activation="relu"))
    
    
    #model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D_8(filters=params[2], kernel_size=(4,4), positions=positions,
                              padding="same", name="Conv8_5", activation="relu"))
    
    model.add(Convolution2D_8(filters=params[2], kernel_size=(4,4), positions=positions,
                              padding="same", name="Conv8_6", activation="relu"))
    
    
    
    #model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
   
    #model.add(Dropout(0.25))
    #model.add(FFT_OUT(name='FFT_OUT'))
    model.add(Flatten())
    model.add(Dense(params[3], activation='relu'))
    model.add(Dense(classes,activation='softmax'))

    return model


def fft_filter_clf(classes, shape=(75,75)):

    model = Sequential()

    model.add(FFT_IN(input_shape=(*shape,3),name='FFT_IN'))
    #model.add(Dropout(0.25))
    #model.add(FFT_OUT(name='FFT_OUT'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(classes,activation='softmax'))

    return model


def fft_filter_clf_rot(classes, shape=(75,75)):

    model = Sequential()

    model.add(FFT_IN(input_shape=(*shape,3),name='FFT_IN'))
    model.add(Rot2D(name="ROT"))
    
    #model.add(Dropout(0.25))
    #model.add(FFT_OUT(name='FFT_OUT'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(classes,activation='softmax'))

    return model




def fft_vgg(classes, shape=(75,75)):
    
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    
    model = Sequential()

    model.add(FFT_IN(input_shape=(*shape,3),name='FFT_IN'))
    

   
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(FFT_IN(name='FFT_IN2'))

    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(FFT_IN(name='FFT_IN3'))

    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(FFT_IN(name='FFT_IN4'))

    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(FFT_IN(name='FFT_IN5'))

    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(FFT_IN(name='FFT_IN6'))
    
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

def fft_vgg_rot(classes, shape=(75,75)):
    
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]
    
    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=(*shape,3)))
    locnet.add(Conv2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Conv2D(20, (5, 5)))
    locnet.add(Flatten())
    locnet.add(Dense(50))
    locnet.add(Activation('relu'))
    #locnet.add(Dense(2,weights=weights, activation="relu")) # angle
    
    locnet.add(Dense(6,weights=weights)) # angle
    
    
    
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    
    model = Sequential()
    #
    #model.add(FFT_IN(input_shape=(*shape,3),name='FFT_IN'))
    #model.add(Rot2D(name="ROT",input_shape=(*shape,3)))
    
    model.add(SpatialTransformer(localization_net=locnet, name="LocNet",
                             output_size=shape, input_shape=(*shape,3)))
    
    model.add(FFT_IN(name='FFT_IN'))
    
    model.add(Convolution2D_8(filters=32, kernel_size=(3, 3), padding="same", name="Conv8"))
    model.add(FFT_IN(name='FFT_IN2'))

    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    #model.add(Rot2D(name="ROT2"))
    
    model.add(FFT_IN(name='FFT_IN3'))

    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(FFT_IN(name='FFT_IN4'))

    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(FFT_IN(name='FFT_IN5'))

    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(FFT_IN(name='FFT_IN6'))
    
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model



def vgg_rot(classes, shape=(75,75), params=[16,32,64,256], kernel_size=(3,3)):
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    
    model = Sequential()
    #
    model.add(FFT_IN(input_shape=(*shape,3),name='FFT_IN'))
    #model.add(Rot2D(name="ROT",input_shape=(*shape,3)))
    
    '''model.add(RotTransformer(localization_net=locnet, name="LocNet",
                             output_size=shape,input_shape=(*shape,3)))'''
    
    #model.add(FFT_IN(name='FFT_IN'))
    
    model.add(Convolution2D_8(filters=params[0], kernel_size=kernel_size, padding="same", name="Conv8", activation="relu"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(params[1], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(params[1], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(params[2], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(params[2], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(params[3]))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))


    # return the constructed network architecture
    return model



def fft_vgg_rot_small(classes, shape=(75,75), params=[16,32,64,256], kernel_size=(3,3)):
    
    # initial weights
    
    
    '''b = np.zeros((2), dtype='float32')
    b[0] = 1
    b[1] = 0
    W = np.zeros((50, 2), dtype='float32')
    weights = [W, b.flatten()]
    
    
    
    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=(*shape,3)))
    locnet.add(Conv2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Conv2D(20, (5, 5)))

    locnet.add(Flatten())
    locnet.add(Dense(50))
    locnet.add(Activation('relu'))
    locnet.add(Dense(2, weights=weights, activation="sigmoid"))
    '''
    
    
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    
    model = Sequential()
    #
    #model.add(FFT_IN(input_shape=(*shape,3),name='FFT_IN'))
    #model.add(Rot2D(name="ROT",input_shape=(*shape,3)))
    
    '''model.add(RotTransformer(localization_net=locnet, name="LocNet",
                             output_size=shape,input_shape=(*shape,3)))'''
    
    #model.add(FFT_IN(name='FFT_IN'))
    
    model.add(Convolution2D_8(filters=params[0], kernel_size=kernel_size,input_shape=(*shape,3), padding="same", name="Conv8", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.25))
    
    #model.add(FFT_IN(name='FFT_IN2'))
    model.add(Convolution2D_8(filters=params[1], kernel_size=(5,5), padding="same", name="Conv8_2", activation="relu"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(params[1], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(params[1], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(params[2], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(params[2], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(params[3]))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))


    # return the constructed network architecture
    return model



if __name__ == '__main__':

    model = fft_filter_clf(2)
    model.summary()