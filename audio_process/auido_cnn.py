import random
import numpy as np
from collections import deque
import scipy.io as sio
import json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, Conv2D, MaxPool2D, TimeDistributed, Merge, Input, UpSampling2D, multiply, add, \
    concatenate, BatchNormalization, AvgPool2D
from keras.optimizers import SGD, Adam
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import os
from keras import layers
import cv2
from random import shuffle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
import matplotlib.pyplot as plt
from keras.applications import VGG16
import glob
from keras.utils import np_utils
from keras import optimizers


def get_data_and_label():
    txt_path = "C:\\PR_project\\split_data_speech\\audio.txt"
    audio = open(txt_path, 'r').read().splitlines()
    file = os.listdir("C:\PR_project\\split_data_speech")
    dict = {}
    count = 0
    for string in file:
        if string != 'audio.txt' and string != 'savemat':
            dict[string] = count
            count += 1
    audio_dict = {}
    Name = []
    for lines in audio:
        splits = lines.split("\\")
        name = splits[-1].split(".")[0]
        label = splits[-2]
        label_num = dict[label]
        audio_dict[name] = label_num
        Name.append(name)

    return Name, audio_dict


def get_audio_mat(audio_dict):
    path_mat = "C:\\PR_project\\split_data_speech\\savemat\\"
    mat_list = os.listdir(path_mat)
    shuffle(mat_list)
    train_list = mat_list[0:int(0.8*len(mat_list))]
    test_list = mat_list[int(0.8*len(mat_list)):-1]

    MAT = []
    MAT_label = []
    for i in range(len(train_list)):
        mat = sio.loadmat(path_mat + mat_list[i])['MFSC']
        mat = mat[:,:,:,0]
        mat = np.array(mat)
        maximum = float(np.max(mat))
        mat /= maximum
        MAT.append(mat)
        name = mat_list[i].split(".")[0]
        MAT_label.append(audio_dict[name])

    MAT = np.array(MAT)
    MAT_label = np_utils.to_categorical(MAT_label, 8)

    np.save("train_data.npy", MAT)
    np.save("train_label.npy", MAT_label)

    MAT = []
    MAT_label = []
    for j in range(len(test_list)):
        mat = sio.loadmat(path_mat + mat_list[i])['MFSC']
        mat = mat[:, :, :, 0]
        MAT.append(mat)
        name = mat_list[i].split(".")[0]
        MAT_label.append(audio_dict[name])

    MAT = np.array(MAT)
    MAT_label = np_utils.to_categorical(MAT_label, 8)

    np.save("test_data.npy",MAT)
    np.save("test_label.npy",MAT_label)

    return MAT, MAT_label


def build_model():
    inp_au = Input(shape=(64, 64, 3))

    '''
    # forward 1 audio
    '''
    conv1_au = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp_au)
    conv2_au = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_au)
    batch1 = BatchNormalization()(conv2_au)
    pool2_au = MaxPool2D(pool_size=(2, 2))(batch1)

    conv3_au = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2_au)
    batch2 = BatchNormalization()(conv3_au)
    pool3_au = MaxPool2D(pool_size=(2, 2))(batch2)

    conv4_au = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3_au)
    batch3 = BatchNormalization()(conv4_au)
    pool4_au = MaxPool2D(pool_size=(2, 2))(batch3)
    #
    # conv5_au = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4_au)
    # pool5_au = MaxPool2D(pool_size=(2, 2))(conv5_au)
    # pool6_au = MaxPool2D(pool_size=(2, 2))(pool5_au)

    flt = Flatten()(pool4_au)
    dense1 = Dense(256, activation='relu')(flt)
    dp1 = Dropout(0.6)(dense1)
    dense2 = Dense(128,activation='relu')(dp1)
    dp2 = Dropout(0.6)(dense2)
    outputs = Dense(8, activation='softmax')(dp2)
    model = Model(input=inp_au, output=outputs, name="baseline_model.h5")

    optimizer = optimizers.Adam(lr=0.000005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":

    model = build_model()
    model.summary()

    name, audio_dict = get_data_and_label()
    train_data = np.load("train_data.npy")
    train_label = np.load("train_label.npy")
    #
    # c = list(zip(train_data, train_label))
    # shuffle(c)
    # data, label = zip(*c)
    # train_data = np.array(data)
    # train_label = np.array(label)
    #
    if os.path.isfile(model.name):
        model.load_weights(model.name)
        print("Load weights successfully")
    else:
        print("No "+model.name)

    # for i in range(10000):
    #     model.fit(np.array(train_data), np.array(train_label),batch_size=4, nb_epoch=10, shuffle=True, verbose=1)
    #     model.save("baseline_model.h5")

    test_data = np.load("test_data.npy")
    test_label = np.load("test_label.npy")

    # aaa = np.load()
    #
    score = model.evaluate(test_data,test_label,batch_size=4)
    print(score[0], score[1])

    # mat = sio.loadmat("C:\\PR_project\\split_data_speech\\savemat\\5.mat")['MFSC']
    # print(audio_dict["5"])
    # mat = mat[:, :, :, 0]
    # mat = np.expand_dims(mat,axis=0)
    # pre = model.predict(mat,batch_size=1)
    # print(np.argmax(pre))