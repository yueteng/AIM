#https://github.com/suyashtrivedi99/Project-Deep-Learning/tree/master/Project_Deep_Learning
#importing all necessary libraries
from keras.models import Sequential    #For initialising the model
from keras.layers import *      #For adding convolutional layer
from keras.layers import MaxPooling2D  #For adding max pooling layer
from keras.layers import Flatten       #For flattening max pooled layer values into a single vector
from keras.layers import Dense, ZeroPadding2D, BatchNormalization,  Activation,Dropout        #For adding layers to NN
from keras.callbacks import ModelCheckpoint, EarlyStopping

import glob             #for accessing all the images
import numpy as np      #for handling the images as numpy arrays
from PIL import Image   #for resizing the images
from keras.optimizers import adam
from sklearn import preprocessing, model_selection as ms  #for splitting data into Training, Cross - Validating, and Testing parts
from keras.preprocessing.image import ImageDataGenerator  #for image augmentation
import h5py                                               #for saving the model
from keras import layers
from keras.models import *                  #for loading the model
import matplotlib.pyplot as plt  #for plotting training and cross validation accuracies vs epochs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def identity_block(input_tensor, kernel_size, filters, stage, block, IMAGE_ORDERING = 'channels_last'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3


    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def one_side_pad(x):
    x = ZeroPadding2D((1, 1))(x)
    x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), IMAGE_ORDERING = 'channels_last'):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
                      strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
def vgg_16():
    img_input = Input(shape=(64,64,3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    o = Dense(1000, activation="relu")(x)

    o4 = f4
    o4 = Flatten()(o4)
    o4 = Dense(1000, activation="relu")(o4)
    o = Add()([o, o4])

    o3 = f3
    o3 = Flatten()(o3)
    o3 = Dense(1000, activation="relu")(o3)
    o = Add()([o, o3])
    o = Dense(1000, activation="relu")(o)

    o2 = f2
    o2 = Flatten()(o2)
    o2 = Dense(1000, activation="relu")(o2)
    o = Add()([o, o2])
    o = Dropout(0.5)(o)
    o = Dense(1000, activation="relu")(o)
    o = Dropout(0.5)(o)
    o = Dense(1, activation="sigmoid")(o)


    model = Model(img_input, o)

    model.compile(optimizer=adam(lr=0.00001), loss='binary_crossentropy',
                  metrics=['accuracy'])  # define optimizer and loss functions as well as required metrics


    return model
def res50_net():
    bn_axis = 3
    img_input = Input(shape=(64,64,3))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x


    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    o = Dense(1000, activation="relu")(x)

    o4 = f4
    o4 = Flatten()(o4)
    o4 = Dense(1000, activation="relu")(o4)
    o = Add()([o, o4])

    o3 = f3
    o3 = Flatten()(o3)
    o3 = Dense(1000, activation="relu")(o3)
    o = Add()([o, o3])
    o = Dense(1000, activation="relu")(o)

    o2 = f2
    o2 = Flatten()(o2)
    o2 = Dense(1000, activation="relu")(o2)
    o = Add()([o, o2])
    o = Dense(1000, activation="relu")(o)

    o = Dense(1, activation="sigmoid")(o)

    model = Model(img_input, o)
    model.compile(optimizer=adam(lr=0.0005), loss='binary_crossentropy',
                  metrics=['accuracy'])  # define optimizer and loss functions as well as required metrics
    return model
def Alexnet():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    levels = []
    img_input = Input(shape=(64,64,3))
    x = (ZeroPadding2D((pad, pad)))(img_input)
    x = (Conv2D(filter_size, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(128, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(256, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    for _ in range(2):
        x = (ZeroPadding2D((pad, pad)))(x)
        x = (Conv2D(1024, (kernel, kernel), padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size)))(x)
        levels.append(x)
    [f1, f2, f3, f4, f5] = levels


    o = Flatten()(x)
    o = Dense(1024, activation="relu")(o)
    o = Dropout(0.5)(o)
    o = Dropout(0.5)(o)
    o = Dense(1, activation="sigmoid")(o)
    model = Model(img_input, o)

    model.compile(optimizer=adam(lr=0.00005), loss='binary_crossentropy',
                  metrics=['accuracy'])  # define optimizer and loss functions as well as required metrics

    return model

def train_model(saveing_dataset, model_name, encoder_net):
    X_train = np.load(saveing_dataset + r'\X_train.npy')
    y_train = np.load(saveing_dataset + r'\y_train.npy')
    X_crossval = np.load(saveing_dataset + r'\X_crossval.npy')
    y_crossval = np.load(saveing_dataset + r'\y_crossval.npy')
    X_test = np.load(saveing_dataset + r'\X_test.npy')
    y_test = np.load(saveing_dataset + r'\y_test.npy')

    encoder_net_class = {"VGG":vgg_16(), "Resnet": res50_net(), "Alex":Alexnet()}

    val_size = X_crossval.shape[0] #cross-validation set size

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       rotation_range = 20,
                                       horizontal_flip = True,
                                       vertical_flip = True,)


    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(X_train,
                                         y_train,
                                         batch_size = 32)

    val_generator = test_datagen.flow(X_crossval,
                                      y_crossval,
                                      batch_size = 16)

    test_generator = test_datagen.flow(X_test,
                                       y_test,
                                       batch_size = 1)

    best_model_path =  model_name

    model = encoder_net_class[encoder_net]
    #print(model.summary())
    early_stopping = EarlyStopping(monitor='val_acc', patience=50, verbose=2, mode='max')
    check_point = ModelCheckpoint(best_model_path, monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    callbacks_list = [check_point, early_stopping]
    #training the model
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = len(X_train) / 32,
                                  epochs = 100,
                                  validation_data = val_generator,
                                  validation_steps = val_size / 16,
                                  callbacks=callbacks_list, verbose=0)

    model = load_model(model_name)


    #obtaining accuracy on test set
    test_acc = model.evaluate_generator(test_generator, steps = len(test_generator))

    print(model.metrics_names)
    print('Test Accuracy Obtained: ')
    print(test_acc[1] * 100, ' %')

    #Plotting Training and Testing accuracies
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.savefig("ROC.png", dpi=600)
    plt.show()

def test_model(saveing_dataset, model_name):

    X_test = np.load(saveing_dataset + r'\X_test.npy')
    y_test = np.load(saveing_dataset + r'\y_test.npy')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow(X_test,
                                       y_test,
                                       batch_size=1)


    model = load_model(model_name)

    # obtaining accuracy on test set
    test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))

    print(model.metrics_names)
    print('Test Accuracy Obtained: ')
    print(test_acc[1] * 100, ' %')











