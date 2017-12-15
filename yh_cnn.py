import os
from keras.layers import Dropout, Conv2D, MaxPool2D
from keras.layers import BatchNormalization, Dense, Flatten
from keras.models import Sequential, optimizers
from keras.preprocessing.image import ImageDataGenerator

TRAIN_DATA_DIR = r'C:\PR_project\yh_face_emotion\train'
TEST_DATA_DIR = r'C:\PR_project\yh_face_emotion\test'
BATCH_SIZE = 256
SAVED_WEIGHT_NAME = 'yh_face_16_16_s.h5'
LR = 0.00001
EPOCH = 100


def face_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
                                     activation='relu', padding='same', input_shape=(128, 128, 3), name='conv1'))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2'))
    model.add(BatchNormalization(name='b1'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv3'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='maxpool2'))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv4'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='maxpool3'))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv5'))
    model.add(BatchNormalization(name='b2'))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv7'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv8'))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv9'))
    model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same', name='conv10'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.6))
    model.add(Dense(4, activation="softmax"))
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
        target_size=(128, 128),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    print(train_generator.class_indices)

    validation_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=(128, 128),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    optimizer = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    if os.path.exists(SAVED_WEIGHT_NAME):
        model.load_weights(SAVED_WEIGHT_NAME)

    for i in range(EPOCH):
        print("starting step {}".format(i))

        model.fit_generator(
                train_generator,
                steps_per_epoch=100,
                epochs=2,
                validation_data=validation_generator,
                validation_steps=20)
        model.save_weights(SAVED_WEIGHT_NAME)

model = face_model()
model.summary()
training(model, LR)
