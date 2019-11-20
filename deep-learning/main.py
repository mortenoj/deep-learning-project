"""json module for parsing json"""
import json
import random
import glob

import numpy as np

from keras.models import Sequential
<<<<<<< Updated upstream
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm
=======
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from tensorflow.python.client import device_lib
import tensorflow as tf

CONST_epochs = 100
CONST_optimizer = "adam"
CONT_activationfunction = "relu"
>>>>>>> Stashed changes

def create_dataset(path):
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

<<<<<<< Updated upstream
def create_model(parameters):
    """Creates a keras model"""

    model = Sequential()
    # model.add(Flatten())
    model.add(
        Dense(
            parameters["neurons"],
            kernel_initializer=parameters["init_mode"],
            activation=parameters["activation"],
            kernel_constraint=maxnorm(parameters["weight_constraint"])
        )
    )

    model.add(Dropout(parameters["dropout_rate"]))

=======
def create_model(optimizer=CONST_optimizer):
    """Creates a keras model"""

    model = Sequential()
    # model.add(keras.layers.Flatten())
    model.add(Dense(256, activation=CONT_activationfunction))
    model.add(Dense(256, activation=CONT_activationfunction))
    #model.add(Dense(64, activation=CONT_activationfunction))
    #model.add(Dense(64, activation=CONT_activationfunction))
>>>>>>> Stashed changes
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


def train_model(train_x, train_y, val_x, val_y):
    """Initializes the training of the model

    Also evaluates the training after run
    """

<<<<<<< Updated upstream
    parameters = {
        "optimizer": "Adam",
        "activation": "relu",
        "init_mode": "normal",
        "learn_rate": 0.01,
        "momentum": 0,
        "neurons": 50,
        "weight_constraint": 1,
        "dropout_rate": 0.2
    }

    model = create_model(parameters)
    model.fit(train_x, train_y, epochs=100, batch_size=60)
=======
    model = create_model()
    model.fit(train_x, train_y, epochs=CONST_epochs, batch_size=1000)
>>>>>>> Stashed changes
    evaluate_training(model, val_x, val_y)


def evaluate_training(model, val_x, val_y):
    """Evaluates the quality of the training"""

    print("\nEvaluating...\n")
    (val_loss, val_acc) = model.evaluate(val_x, val_y)

    print("evaluated loss: ", val_loss)
    print("evaluated accuracy: ", val_acc)
    print("\n\n")

    correct = 0
    for i, pred in enumerate(model.predict(val_x)):
        if np.argmax(pred) == val_y[i]:
            correct += 1

    print("Actual accurancy: ", correct / len(val_x))

def main():
    """The main function"""
    #tensorflow.test.is_built_with_cuda()
    #K.tensorflow_backend._get_available_gpus()
    #print(device_lib.list_local_devices())

    # Create some tensors
    (train_x, train_y) = create_dataset("../dataset/output/*.json")
    (val_x, val_y) = create_dataset("../dataset/output/testset/*.json")

    train_model(train_x, train_y, val_x, val_y)
<<<<<<< Updated upstream
=======
    # parameter_optimization(train_x, train_y)
    # parameter_optimization(val_x, val_y)
>>>>>>> Stashed changes

if __name__ == "__main__":
    main()
