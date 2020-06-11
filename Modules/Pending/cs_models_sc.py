from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Add, LeakyReLU, \
                         MaxPooling2D, concatenate, UpSampling2D,\
                         Multiply
from keras.layers import ZeroPadding2D, Cropping2D

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

 
def fft_layer(image):
    """
    Input: 2-channel array representing image domain complex data
    Output: 2-channel array representing k-space complex data
    """

    # get real and imaginary portions
    real = Lambda(lambda image: image[:, :, :, 0])(image)
    imag = Lambda(lambda image: image[:, :, :, 1])(image)

    image_complex = K.tf.complex(real, imag)  # Make complex-valued tensor
    kspace_complex = K.tf.fft2d(image_complex)

    # expand channels to tensorflow/keras format
    real = K.tf.expand_dims(K.tf.real(kspace_complex), -1)
    imag = K.tf.expand_dims(K.tf.imag(kspace_complex), -1)

    # generate 2-channel representation of k-space
    kspace = K.tf.concat([real, imag], -1)
    return kspace


def ifft_layer(kspace_2channel):
    """
    Input: 2-channel array representing k-space
    Output: 2-channel array representing image domain
    """
    #get real and imaginary portions
    real = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,0])(kspace_2channel)
    imag = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,1])(kspace_2channel)
    
    kspace_complex = K.tf.complex(real,imag) # Make complex-valued tensor
    image_complex = K.tf.ifft2d(kspace_complex)
    
    # expand channels to tensorflow/keras format
    real = K.tf.expand_dims(K.tf.real(image_complex),-1)
    imag = K.tf.expand_dims(K.tf.imag(image_complex),-1)
    
    # generate 2-channel representation of image domain
    image_complex_2channel = K.tf.concat([real, imag], -1)
    return image_complex_2channel

def cnn_block(cnn_input, depth, nf, kshape,channels):
    """
    :param cnn_input: Input layer to CNN block
    :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
    2 channels
    :param nf: Number of filters of convolutional layers, except for the last
    :param kshape: Shape of the convolutional kernel
    :return: 2-channel, complex reconstruction
    """
    layers = [cnn_input]

    for ii in range(depth):
        # Add convolutional block
        layers.append(Conv2D(nf, kshape, padding='same')(layers[-1]))
        layers.append(LeakyReLU(alpha=0.1)(layers[-1]))
    final_conv = Conv2D(channels, (1, 1), activation='linear')(layers[-1])
    rec1 = Add()([final_conv,cnn_input])
    return rec1


def unet_block(unet_input, kshape=(3, 3),channels = 2):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """

    conv1 = Conv2D(48, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same',kernel_regularizer=l1_reg)(conv7)

    conv8 = Conv2D(channels, (1, 1), activation='linear',kernel_regularizer=l1_reg)(conv7)
    out = Add()([conv8, unet_input])
    return out

def unet_blockv2(unet_input, kshape=(3, 3),channels = 2):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(channels, (1, 1), activation='linear')(conv7)
    return conv8

def DC_block(rec,mask,sampled_kspace,channels,kspace = False):
    """
    :param rec: Reconstructed data, can be k-space or image domain
    :param mask: undersampling mask
    :param sampled_kspace:
    :param kspace: Boolean, if true, the input is k-space, if false it is image domain
    :return: k-space after data consistency
    """

    if kspace:
        rec_kspace = rec
    else:
        rec_kspace = Lambda(fft_layer)(rec)
    rec_kspace_dc =  Multiply()([rec_kspace,mask])
    rec_kspace_dc = Add()([rec_kspace_dc,sampled_kspace])
    return rec_kspace_dc

def deep_cascade_flat_unrolled(depth_str = 'ikikii', H=256,W=256,depth = 5,kshape = (3,3), nf = 48,channels = 2):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image heigh
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:
        
        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(cnn_block(layers[-1],depth,nf,kshape,channels))

        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,channels,kspace=kspace_flag))
        kspace_flag = True
    out = Lambda(ifft_layer)(layers[-1])
    model = Model(inputs=[inputs,mask], outputs=out)
    return model

def deep_cascade_unet(depth_str='ki', H=218, W=170, Hpad = 3, Wpad = 3, kshape=(3, 3),channels = 2):

    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:
        
        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(ZeroPadding2D(padding=(Hpad,Wpad))(layers[-1]))
        layers.append(unet_block(layers[-1], kshape,channels))
        layers.append(Cropping2D(cropping=(Hpad,Wpad))(layers[-1]))
        
        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,channels,kspace=kspace_flag))
        kspace_flag = True
    out = Lambda(ifft_layer)(layers[-1])
    model = Model(inputs=[inputs,mask], outputs=out)
    return model

def tf_ifft2d(H,W,channels):
    inputs = Input(shape=(H,W,channels))
    out = Lambda(ifft_layer)(inputs)
    model = Model(inputs=inputs, outputs=out)
    return model

def enhancement_unet(H=218, W=170, Hpad = 3, Wpad = 3):
    inputs = Input(shape=(H,W,2))
    inputs2 = Input(shape=(H,W,1))
    inputs_zp = ZeroPadding2D(padding=(Hpad,Wpad))(inputs)
    out = unet_blockv2(inputs_zp,channels = 1)
    out_cp = Cropping2D(cropping=(Hpad,Wpad))(out)
    out2 = Add()([out_cp, inputs2])
    model = Model(inputs=[inputs,inputs2], outputs=out2)
    return model


