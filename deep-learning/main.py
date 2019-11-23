"""json module for parsing json"""
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import random
import glob
import numpy as np
import time
import tensorflow as tf
import testcases

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.callbacks import TensorBoard
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers




CONST_NAME = "CSGO_Learning_adam_relu_96nodes_3layers_150epochs{}".format(int(time.time()))

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def create_dataset(path, testcase = 0):
    """Creates a dataset from the json data in path"""

    t_max_worth = 42750
    ct_max_worth = 44750
    dataset = []

    for filename in glob.glob(path):
        with open(filename) as json_file:
            data = json.load(json_file)
            for match_round in data:
                if match_round["CTEquipment"] is None or match_round["TEquipment"] is None:
                    continue
                if not testcase == 0:
                    if testcases.filter_testcase(testcase, match_round) is False:
                        continue
                
                m_round = []
                m_round.append(match_round["CTSideTotalWorth"] / ct_max_worth)

                for item in match_round["CTEquipment"]:
                    m_round.append(item["Count"] / 5)

                m_round.append(match_round["TSideTotalWorth"] / t_max_worth)
                for item in match_round["TEquipment"]:
                    m_round.append(item["Count"] / 5)

                m_round.append(get_map_id(match_round["Map"]) / 9) # Maybe not divide by 9 as we don't necessarily need to

                dataset.append([m_round, match_round["TerroristsWon"]])

    random.shuffle(dataset)
    return process_dataset(dataset)

def get_map_id(map_name):
    if map_name == "de_dust2":
        return 1
    elif map_name == "de_nuke":
        return 2
    elif map_name == "de_cache":
        return 3
    elif map_name == "de_overpass":
        return 4
    elif map_name == "de_train":
        return 5
    elif map_name == "de_vertigo":
        return 6
    elif map_name == "de_mirage":
        return 7
    elif map_name == "de_inferno":
        return 8
    elif map_name == "de_cbble":
        return 9
    else:
        return 0 #unknown


def process_dataset(dataset):
    """Processess a dataset.

    Returns a touple with separate lists for raw data and labels.
    This fits the format required for deep learning
    """

    data_x = []
    data_y = []
    for features, label in dataset:
        data_x.append(features)
        data_y.append(label)

    return (np.array(data_x), np.array(data_y))

def create_model(parameters):
    """Creates a keras model"""

    model = Sequential()
    # model.add(Flatten())

    for _ in range(0, parameters["layer_count"]):
        model.add(
            Dense(
                parameters["neurons"],
                kernel_initializer=parameters["init_mode"],
                activation=parameters["activation"],
                kernel_constraint=maxnorm(parameters["weight_constraint"])
            )
        )

    model.add(Dropout(parameters["dropout_rate"]))

    model.add(Dense(2, activation='softmax'))

    optimizer = parameters["optimizer"]

    if optimizer == "sgd":
        optimizer = SGD(lr=parameters["learn_rate"], momentum=parameters["momentum"])


    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def evaluate_training(model, val_x, val_y):
    """Evaluates the quality of the training"""

    print("\nEvaluating...\n")
    (val_loss, val_acc) = model.evaluate(val_x, val_y, verbose = 0)

    print("evaluated loss: ", val_loss)
    print("evaluated accuracy: ", val_acc)
    print("\n\n")

    correct = 0
    for i, pred in enumerate(model.predict(val_x)):
        if np.argmax(pred) == val_y[i]:
            correct += 1

    print("Actual accuracy: ", correct / len(val_x))

def train_different_parameters(train_x, train_y):
    dense_layers = [3]
    layer_sizes = [32, 64, 128]
    optimizers = ["adam", "nadam", "adadelta", "sgd", "adagrad", "RMSprop"]
    activation_functions = ["relu", "elu", "selu", "sigmoid"]

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for optimizer in optimizers:
                for activation_function in activation_functions:
                    NAME = "{}-optimizer-{}-activation-{}-nodes-{}-layers-{}".format(optimizer, activation_function, layer_size, dense_layer, int(time.time()))
                    
                    parameters = {
                        "optimizer": optimizer,
                        "activation": activation_function,
                        "init_mode": "normal",
                        "learn_rate": 0.01,
                        "momentum": 0,
                        "neurons": layer_size,
                        "layer_count": dense_layer,
                        "weight_constraint": 1,
                        "dropout_rate": 0.2
                    }

                    model = create_model(parameters)
                    tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))
                    model.fit(train_x, train_y, epochs=4000, batch_size=4096, verbose=0, validation_split=0.2, callbacks=[tensorboard])

def train_model(train_x, train_y):
    """Initializes the training of the model

    Also evaluates the training after run
    """

    parameters = {
        "optimizer": "adam",
        "activation": "relu",
        "init_mode": "normal",
        "learn_rate": 0.01,
        "momentum": 0,
        "neurons": 96,
        "layer_count": 3,
        "weight_constraint": 1,
        "dropout_rate": 0.2 #0.2
    }

    model = create_model(parameters)
    tensorboard = TensorBoard(log_dir="logs\{}".format(CONST_NAME))
    model.fit(train_x, train_y, epochs=150, batch_size=4096, callbacks=[tensorboard]) #validation_split=0.2,
    return model




def main():
    """The main function"""
    #tensorflow.test.is_built_with_cuda()
    #K.tensorflow_backend._get_available_gpus()
    #print(device_lib.list_local_devices())

    # Create some tensors
    #train_different_parameters(train_x, train_y)

    # Create data sets
    (train_x, train_y) = create_dataset("../dataset/output/*.json")
    (val_x, val_y) = create_dataset("../dataset/output/testset/*.json")

    # Train and evaluate
    # model = train_model(train_x, train_y)
    # evaluate_training(model, val_x, val_y)

    # # Save to file
    # print("\nSaving model to file...\n")
    # model.save("models/adam_relu_96neurons_3layers_150epochs.h5")

    # Load from file
    print("\nLoading model from file...\n")  
    new_model = keras.models.load_model("models/adam_relu_96neurons_3layers_150epochs.h5")
    print("\nRunning validation set...\n")  
    evaluate_training(new_model, val_x, val_y)

    for i in range(1, 10):
        print("\nRunning test case nr {}...\n".format(i))  
        (test_x, test_y) = create_dataset("../dataset/output/testset/*.json", i)
        evaluate_training(new_model, test_x, test_y)
        print("-------------------------------------------")

if __name__ == "__main__":
    main()
