import keras.backend as K
import os
from keras.layers import Activation, Input, Dropout, Conv2D, UpSampling2D, MaxPool2D
from keras.layers import LeakyReLU, BatchNormalization, Dense
from keras.models import Model, Sequential, optimizers
from keras.preprocessing.image import ImageDataGenerator


TRAIN_DATA_DIR = r'E:\DATASET\pr\face_emotion\train'
TEST_DATA_DIR = r'E:\DATASET\pr\face_emotion\test'
BATCH_SIZE = 64
SAVED_WEIGHT_NAME = 'face.h5'
LR = 0.0001
EPOCH = 100  # epoch * 2 is correct

def face_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(2, 2),
                                     activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dense(2048))
    model.add(Dropout(0.6))
    model.add(Dense(512))
    model.add(Dropout(0.6))
    model.add(Dense(8, activation="softmax"))
    return model


def training(model, LR):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        # vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2
        )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    print(train_generator.class_indices)

    validation_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    optimizer = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    for i in range(EPOCH):
        print("starting step {}".format(i))
        if os.path.exists(SAVED_WEIGHT_NAME):
            model.load_weights(SAVED_WEIGHT_NAME)
        model.fit_generator(
                train_generator,
                steps_per_epoch=250,
                epochs=2,
                validation_data=validation_generator,
                validation_steps=50)
        model.save_weights(SAVED_WEIGHT_NAME)

training(face_model(), LR)