import os
import warnings
import random
import shutil
from shutil import copyfile
import data_paths as dp
import pandas as pd
import numpy as np
from keras.utils import load_img, img_to_array

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings("ignore")
from colorama import Fore


class weather:
    def __init__(self, path, base_dir):

        self.base_dir = base_dir
        self.path = path
        self.height = None
        self.width = None
        self.batch_size = None
        self.model = None
        self.history = None
        self.best_model = None
        self.valid_generator = None
        self.train_generator = None

    def load_dataset(self):
        self.path = os.listdir(self.path)

    def new_dir(self):
        os.mkdir(self.base_dir)
        print(Fore.GREEN + "New base directory created.")

    def create_folder(self):

        self.train_dir = os.path.join(self.base_dir, "train")
        os.mkdir(self.train_dir)

        self.validation_dir = os.path.join(self.base_dir, "validation")
        os.mkdir(self.validation_dir)

        self.train_cloud_dir = os.path.join(self.train_dir, 'cloudy')
        os.mkdir(self.train_cloud_dir)

        self.train_foggy_dir = os.path.join(self.train_dir, 'foggy')
        os.mkdir(self.train_foggy_dir)

        self.train_rainy_dir = os.path.join(self.train_dir, 'rainy')
        os.mkdir(self.train_rainy_dir)

        self.train_shine_dir = os.path.join(self.train_dir, 'shine')
        os.mkdir(self.train_shine_dir)

        self.train_sunrise_dir = os.path.join(self.train_dir, 'sunrise')
        os.mkdir(self.train_sunrise_dir)

        self.validation_cloud_dir = os.path.join(self.validation_dir, 'cloudy')
        os.mkdir(self.validation_cloud_dir)

        self.validation_foggy_dir = os.path.join(self.validation_dir, 'foggy')
        os.mkdir(self.validation_foggy_dir)

        self.validation_rainy_dir = os.path.join(self.validation_dir, 'rainy')
        os.mkdir(self.validation_rainy_dir)

        self.validation_shine_dir = os.path.join(self.validation_dir, 'shine')
        os.mkdir(self.validation_shine_dir)

        self.validation_sunrise_dir = os.path.join(self.validation_dir, 'sunrise')
        os.mkdir(self.validation_sunrise_dir)

        print(Fore.GREEN + "New folders created")

    def control_files(self):
        try:
            if os.path.exists(self.train_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.train_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.train_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.validation_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.validation_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.validation_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.train_cloud_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.train_cloud_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.train_cloud_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.train_foggy_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.train_foggy_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.train_foggy_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.train_rainy_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.train_rainy_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.train_rainy_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.train_shine_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.train_shine_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.train_shine_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.train_sunrise_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.train_sunrise_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.train_sunrise_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.validation_cloud_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.validation_cloud_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.validation_cloud_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.validation_foggy_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.validation_foggy_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.validation_foggy_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.validation_rainy_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.validation_rainy_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.validation_rainy_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.validation_shine_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.validation_shine_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.validation_shine_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

        try:
            if os.path.exists(self.validation_sunrise_dir):
                print(Fore.LIGHTCYAN_EX + "File already exists!")
            else:
                with open(self.validation_sunrise_dir, "w") as file:
                    file.write(Fore.LIGHTYELLOW_EX + "File is created.")
            with open(self.validation_sunrise_dir, "r") as file:
                content = file.read()

        except Exception as e:
            print(Fore.LIGHTRED_EX + "::Error::", str(e))

    def split_data(self, source, training, validation, split_size):
        files = []
        for filename in os.listdir(source):
            file = source + filename
            if os.path.getsize(file) > 0:
                files.append(filename)
            else:
                print(filename + Fore.RED + " is zero length, so ignoring.")

        training_length = int(len(files) * split_size)
        valid_length = int(len(files) - training_length)
        shuffled_set = random.sample(files, len(files))
        training_set = shuffled_set[0:training_length]
        valid_set = shuffled_set[valid_length:]

        for filename in training_set:
            this_file = source + filename
            destination = training + filename
            copyfile(this_file, destination)

        for filename in valid_set:
            this_file = source + filename
            destination = validation + filename
            copyfile(this_file, destination)

    def show_new_dataset(self):
        global train_data, valid_data
        for i in ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']:
            print(Fore.MAGENTA + 'Training {} images are: '.format(i) + str(
                len(os.listdir(train_data + i + '\\'))))

        for i in ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']:
            print(Fore.YELLOW +
                  'Valid {} images are: '.format(i) + str(
                len(os.listdir(valid_data + i + '\\'))))

    def data_augmentation(self):
        print(Fore.LIGHTWHITE_EX)
        global train_data
        global valid_data
        self.width = 256
        self.height = 256
        self.batch_size = 16
        train_datagen = ImageDataGenerator(rescale=1 / 255.0,
                                           rotation_range=40,
                                           zoom_range=0.3,
                                           horizontal_flip=True,
                                           width_shift_range=30,
                                           fill_mode="nearest")
        self.train_generator = train_datagen.flow_from_directory(train_data,
                                                                 batch_size=self.batch_size,
                                                                 class_mode="categorical",
                                                                 target_size=(self.width, self.height))

        valid_datagen = ImageDataGenerator(rescale=1 / 255.0)
        self.valid_generator = valid_datagen.flow_from_directory(valid_data,
                                                                 batch_size=self.batch_size,
                                                                 class_mode="categorical",
                                                                 target_size=(self.width, self.height))

    def create_model(self):
        self.model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(self.height, self.width, 3)), MaxPooling2D(2, 2),
            Conv2D(16, 3, padding='same', activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(32, 3, padding='same', activation='relu'),
            Conv2D(32, 3, padding='same', activation='relu'),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, 3, padding='same', activation='relu'),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(5, activation='softmax')
        ])

        self.model.summary()

    def mod_compile(self):
        callbacks = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="min")
        best_model_file = "C:\\Users\\hp\\Desktop\\weatherdataset\\CNN_aug_best_weights4.h5"
        best_model = ModelCheckpoint(best_model_file, monitor="val_accuracy", verbose=1, save_best_only=True,
                                     mode="max")

        self.model.compile(optimizer="Adam",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

        self.history = self.model.fit(self.train_generator,
                                      epochs=10,
                                      verbose=1,
                                      validation_data=self.valid_generator,
                                      callbacks=[best_model])

        return self.model, self.history

    def visualize(self):
        acc = self.history.history["accuracy"]
        val_acc = self.history.history["val_accuracy"]
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]

        epochs = range(len(acc))

        fig = plt.figure(figsize=(14, 7))
        plt.plot(epochs, acc, 'r', label="Training Accuracy")
        plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and validation accuracy")
        plt.legend(loc="lower right")
        plt.savefig("accuracy_graph4.png")
        plt.show()

        fig2 = plt.figure(figsize=(14, 7))
        plt.plot(epochs, loss, 'r', label="Training Loss")
        plt.plot(epochs, val_loss, 'b', label="Validation Loss")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("loss_graph4.png")
        plt.title("Training and validation loss")

    def preprocess_image(self, path):
        # Test Performance on Test Data
        img = load_img(path, target_size=(self.height, self.width))
        a = img_to_array(img)
        a = np.expand_dims(a, axis=0)
        a /= 255.
        return a


dataset = "C:\\Users\\hp\\Desktop\\weatherdataset"
base = 'C:\\Users\\hp\\Desktop\\weatherdataset\\weather_pred4'
split_size = .85
train_data = 'C:\\Users\\hp\\Desktop\\weatherdataset\\weather_pred4\\train\\'
valid_data = 'C:\\Users\\hp\\Desktop\\weatherdataset\\weather_pred4\\validation\\'

weatherrr_ = weather(dataset, base)
weatherrr_.load_dataset()
weatherrr_.new_dir()
weatherrr_.create_folder()
weatherrr_.control_files()
weatherrr_.split_data(dp.CLOUDY_SOURCE_DIR, dp.TRAINING_CLOUDY_DIR, dp.VALID_CLOUDY_DIR, split_size)
weatherrr_.split_data(dp.FOGGY_SOURCE_DIR, dp.TRAINING_FOGGY_DIR, dp.VALID_FOGGY_DIR, split_size)
weatherrr_.split_data(dp.RAINY_SOURCE_DIR, dp.TRAINING_RAINY_DIR, dp.VALID_RAINY_DIR, split_size)
weatherrr_.split_data(dp.SHINE_SOURCE_DIR, dp.TRAINING_SHINE_DIR, dp.VALID_SHINE_DIR, split_size)
weatherrr_.split_data(dp.SUNRISE_SOURCE_DIR, dp.TRAINING_SUNRISE_DIR, dp.VALID_SUNRISE_DIR, split_size)
weatherrr_.show_new_dataset()
weatherrr_.data_augmentation()
weatherrr_.create_model()
weatherrr_.mod_compile()
weatherrr_.visualize()

test_images_dir = 'C:\\Users\\hp\\Desktop\\weatherdataset\\tests\\alien_test\\'
test_df = pd.read_csv('C:\\Users\\hp\\Desktop\\weatherdataset\\tests\\test.csv')

# put them in a list
test_dfToList = test_df['Image_id'].tolist()
test_ids = [str(item) for item in test_dfToList]

test_images = [test_images_dir + item for item in test_ids]
test_preprocessed_images = np.vstack([weatherrr_.preprocess_image(fn) for fn in test_images])
np.save('C:\\Users\\hp\\Desktop\\weatherdataset\\test_preproc4_CNN.npy', test_preprocessed_images)
