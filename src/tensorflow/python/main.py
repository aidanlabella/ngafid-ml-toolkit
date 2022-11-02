#!/usr/bin/env python3 

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

classes = {
    "Stall-like_Event": 1,
    "False_Positive": 2,
    "False_Positive_(forward_slip)": 3,
}


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def load_ngafid(dir, filename):
    data = np.loadtxt(dir + filename, delimiter=",")
    label = filename.split(".")[1]
    return np.array(data), classes[label]

# file_dir = "/Users/aidan/RIT/UGRA/D2S2/extracted_loci_events/oct16"
# file_dir = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
file_dir = "/Users/aidan/RIT/UGRA/D2S2/extracted_loci_events/nov22/"

# x_train, y_train = readucr(file_dir + "FordA_TRAIN.tsv")
# x_test, y_test = readucr(file_dir + "FordA_TEST.tsv")
x_train = np.array([])
y_train = np.array([])

x_arr = []
y_arr = []

for file in os.listdir(file_dir):
    if not os.path.isdir(file):
        print(file)
        x, y = load_ngafid(file_dir, file)
        x_arr.append(x)
        y_arr.append(y)
    
x_train = np.stack(x_arr, axis=0)
print(x_train)
y_train = np.stack(y_arr, axis=0)
print(y_train)

# file_dir = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
# x_train, y_train = readucr(file_dir + "FordA_TRAIN.tsv")
# print(x_train)
# print(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

print("shape:")
print(x_train.shape[0])
print(x_train.shape[1])
print(x_train.shape[2])

n_classes = 3
classes = np.unique(y_train)
num_classes = len(classes)
print(classes)
plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
# plt.show()
# print()

# num_classes = len(np.unique(y_train))
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
y_train[y_train == -1] = 0
# y_test[y_test == -1] = 0

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)

epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0018
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)
