# coding:utf-8
#https://github.com/scikit-learn-contrib/imbalanced-learn.git
import argparse
import os
import numpy as np
from keras.optimizers import adam
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
from keras.models import Model, load_model
import glob
from PIL import Image
import random
import cv2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import roc_curve, auc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_parameters():
    parser = argparse.ArgumentParser(description="step_3")
    parser.add_argument('--image_Neg', type=str, default='Image_concat/Negtivate')
    parser.add_argument('--image_Pos', type=str, default='Image_concat/Positivate')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--data', type=str, default="dataset")
    parser.add_argument("--train_percentage", type=float, default=0.6)

    config = parser.parse_args()

    return config



def load_data(config):
    dataset = config.image_Pos
    X, Y = list(), list()

    files = glob.glob(dataset + '/*.jpg')
    for file in files:
        img = np.array(Image.open(file))
        X.append(img)
        Y.append(1)

    dataset = config.image_Neg
    files = glob.glob(dataset + '/*.jpg')
    for file in files:
        img = np.array(Image.open(file))
        X.append(img)
        Y.append(0)
    return X, Y


def split_data(config):
    split_path = config.data
    X, Y = load_data(config)
    index = np.arange(len(Y))
    np.random.shuffle(index)

    tran_len = int(np.round(len(Y) * config.train_percentage))
    valid_len = int(tran_len + np.round((len(Y) - tran_len) / 2))
    train_index = index[:tran_len]
    valid_index = index[tran_len:valid_len]
    test_index = index[valid_len:]
    X, Y = np.array(X), np.array(Y)
    Train_X = X[train_index]
    Train_Y = Y[train_index]

    Valid_X = X[valid_index]
    Valid_Y = Y[valid_index]

    Test_X = X[test_index]
    Test_Y = Y[test_index]

    np.save(os.path.join(split_path, 'train_data.npy'), Train_X)
    np.save(os.path.join(split_path, 'train_label.npy'), Train_Y)
    np.save(os.path.join(split_path, 'valid_data.npy'), Valid_X)
    np.save(os.path.join(split_path, 'valid_label.npy'), Valid_Y)
    np.save(os.path.join(split_path, 'test_data.npy'), Test_X)
    np.save(os.path.join(split_path, 'test_label.npy'), Test_Y)


def load_dataset(config):
    train_X = np.load(os.path.join(config.data, "train_data.npy"))
    train_Y = np.load(os.path.join(config.data, "train_label.npy"))

    valid_X = np.load(os.path.join(config.data, "valid_data.npy"))
    valid_Y = np.load(os.path.join(config.data, "valid_label.npy"))

    test_X = np.load(os.path.join(config.data, "test_data.npy"))
    test_Y = np.load(os.path.join(config.data, "test_label.npy"))

    data = {"Train_X": train_X, "Train_Y": train_Y,
            "Valid_X": valid_X, "Valid_Y": valid_Y,
            "Test_X": test_X, "Test_Y": test_Y}

    return data


def fc_model():
    input = Input(shape=(64*5, 64 * 5, 3))

    x = Conv2D(128, (3, 3), activation="relu")(input)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)


    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)

    #
    # x = Conv2D(128, (3, 3))(x)
    #
    # x = Conv2D(64, (3, 3))(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    # x = Dense(64, activation="relu")(x)
    y = Dense(1, activation="sigmoid")(x)

    model = Model(input=input, output=y)
    model.compile(optimizer=adam(lr=0.001), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model():
    model_save_path = "best_model.h5"
    model = fc_model()
    early_stopping = EarlyStopping(monitor='val_acc', patience=100, verbose=0, mode='max')
    check_point = ModelCheckpoint(model_save_path, monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    callbacks_list = [check_point, early_stopping]

    # training the model

    model.fit(x=data['Train_X'], y=data["Train_Y"], epochs=2000, batch_size=6,
              validation_data=(data['Valid_X'], data["Valid_Y"]), callbacks=callbacks_list, verbose=2)


def roc_lists(y_test, test_predict, model):
    y_test_int = np.round(test_predict).astype("int32")
    report = classification_report_imbalanced(y_test, y_test_int, digits=5)
    print(report)

    plt.figure(figsize=(10, 10))

    fpr, tpr, threshold = roc_curve(y_test, test_predict)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)
    lw = 2
    plt.plot(fpr, tpr, color="r",
             lw=lw, label=model + ' (AUC = %0.5f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Compare the ROC of the base model')
    plt.title(model)
    # plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(model + "_ROC.png", dpi=600)
    plt.show()

if __name__ == "__main__":

    config = set_parameters()
    print(config)

    data = load_dataset(config)

    train_model()
