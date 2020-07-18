import glob             #for accessing all the images
import numpy as np      #for handling the images as numpy arrays
from PIL import Image   #for resizing the images

from sklearn import model_selection as ms  #for splitting data into Training, Cross - Validating, and Testing parts
from keras.preprocessing.image import ImageDataGenerator  #for image augmentation
from buliding_model.check_and_creat__file import mkdir, check_folder_contents
from random import shuffle

def train_img_intensification(data, file):
    mkdir(file)
    check_folder_contents(file)
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       rotation_range = 20,
                                       horizontal_flip = True,
                                       vertical_flip = True,)


    i = 0
    for batch in train_datagen.flow(x=data,
                                    batch_size=3000,
                                    save_to_dir=file,
                                    save_prefix=file,
                                    save_format='png'):

        i += 1
        if i > 9:
            break

def read_data(Images_file):
    Images = glob.glob(Images_file + r'\*.png')
    shuffle(Images)

    Images = Images[0:10000]
    data = []
    for file in Images:  # resizing all positive images and converting them to numpy arrays
        img = Image.open(file)
        img_resized = img.resize((64, 64))
        img_array = np.asarray(img_resized)
        data.append(img_array)
    data = np.array(data)
    m = data.shape[0]
    rand_idx = np.arange(m)  # generating indices
    np.random.shuffle(rand_idx)  # randomising indices
    data = data[rand_idx]
    return data

def data_shuffle(x, y):
    y = np.array(y)
    m = x.shape[0]
    rand_idx = np.arange(m)  # generating indices
    np.random.shuffle(rand_idx)  # randomising indices
    x = x[rand_idx]
    y = y[rand_idx]
    return x, y

def split_data(Images_file):
    data = read_data(Images_file)
    X_train, X_new = ms.train_test_split(data, test_size=0.2, random_state=0)
    X_crossval, X_test = ms.train_test_split(X_new, test_size=0.5, random_state=0)
    return X_train, X_crossval, X_test

def create_dataset(pos_dataset, neg_dataset, saving_dataset):

    pos_train, pos_crossval, pos_test = split_data(pos_dataset)
    neg_train, neg_crossval, neg_test = split_data(neg_dataset)

    x_train = np.array(list(pos_train) + list(neg_train))
    x_crossval = np.array(list(pos_crossval) + list(neg_crossval))
    x_test = np.array(list(pos_test) + list(neg_test))
    y_train = np.shape(pos_train)[0]*["1"] + np.shape(neg_train)[0]*["0"]
    y_crossval = np.shape(pos_crossval)[0]*["1"] + np.shape(neg_crossval)[0]*["0"]
    y_test = np.shape(pos_test)[0] * ["1"] + np.shape(neg_test)[0] * ["0"]

    x_train, y_train = data_shuffle(x_train, y_train)
    x_crossval, y_crossval = data_shuffle(x_crossval, y_crossval)
    x_test, y_test = data_shuffle(x_test, y_test)

    mkdir(saving_dataset)
    check_folder_contents(saving_dataset)
    np.save(saving_dataset + r'\X_train.npy', x_train)
    np.save(saving_dataset + r'\y_train.npy', y_train)
    np.save(saving_dataset + r'\X_crossval.npy', x_crossval)
    np.save(saving_dataset + r'\y_crossval.npy', y_crossval)
    np.save(saving_dataset + r'\X_test.npy', x_test)
    np.save(saving_dataset + r'\y_test.npy', y_test)

def main():
    create_dataset("Parasitized", "Uninfected", "data")


if __name__ == "__main__":
    main()