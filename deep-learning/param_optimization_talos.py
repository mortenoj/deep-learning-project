"""json module for parsing json"""
import json
import random
import glob

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, Nadam
from keras.activations import softmax
from keras.losses import categorical_crossentropy, logcosh

import talos
from talos.utils import lr_normalizer

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

                dataset.append([m_round, match_round["TerroristsWon"]])

    random.shuffle(dataset)
    return process_dataset(dataset)

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

def simple_model(x_train, y_train, x_val, y_val, params):
    """Creates a keras model"""
    model = Sequential()
    model.add(
        Dense(
            params['first_neuron'],
            input_dim=90,
            activation=params['activation']
        )
    )

    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    out = model.fit(
        x=x_train,
        y=y_train,
        validation_data=[x_val, y_val],
        epochs=100,
        batch_size=params['batch_size'],
        verbose=0
    )
    return out, model

def concise_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(
        Dense(
            params['first_neuron'],
            input_dim=x_train.shape[1],
            activation=params['activation'],
            kernel_initializer=params['kernel_initializer']
        )
    )

    model.add(Dropout(params['dropout']))

    model.add(
        Dense(
            1,
            activation=params['last_activation'],
            kernel_initializer=params['kernel_initializer']
        )
    )
    model.compile(
        loss=params['losses'],
        optimizer=params['optimizer'],
        metrics=['acc', talos.utils.metrics.f1score]
    )
    history = model.fit(
        x_train,
        y_train, 
        validation_data=[x_val, y_val],
        batch_size=params['batch_size'],
        callbacks=[talos.utils.live()],
        epochs=params['epochs'],
        verbose=0
    )
    return history, model

def comprehensive_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(
        Dense(
            params['first_neuron'],
            input_dim=90,
            activation='relu'
        )
    )

    model.add(Dropout(params['dropout']))
    model.add(
        Dense(y_train.shape[1], activation=params['last_activation']))

    model.compile(
        optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
        loss=params['loss'],
        metrics=['acc']
    )

    out = model.fit(
        x_train,
        y_train,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        verbose=0,
        validation_data=[x_val, y_val]
    )

    return out, model

def parameter_optimization(x_data, y_data):
    """Find the optimal parameters for DL training"""

    # scan_object = simple_analysis(x_data, y_data)
    scan_object = concise_analysis(x_data, y_data)
    # scan_object = comprehensive_analysis(x_data, y_data)

    evaluate_object = talos.Evaluate(scan_object)
    evaluate_object.evaluate(x_data, y_data, folds=10, metric='val_accuracy', task='multi_label')

    return scan_object

def comprehensive_analysis(x_data, y_data):
    params = {
        'lr': (0.1, 10, 10),
        'first_neuron':[4, 8, 16, 32, 64, 128],
        'batch_size': [2, 3, 4],
        'epochs': [200],
        'dropout': (0, 0.40, 10),
        'optimizer': [Adam, Nadam],
        'loss': ['categorical_crossentropy'],
        'last_activation': ['softmax'],
        'weight_regulizer': [None]
    }

    return talos.Scan(
        x=x_data,
        y=y_data,
        params=params,
        model=comprehensive_model,
        experiment_name="comprehensive_analysis",
        fraction_limit=.001
    )

def concise_analysis(x_data, y_data):
    params = {
        'first_neuron':[9, 10, 11],
        'hidden_layers':[0, 1, 2],
        'batch_size': [30],
        'epochs': [100],
        'dropout': [0],
        'kernel_initializer': ['uniform', 'normal'],
        'optimizer': ['Nadam', 'Adam'],
        'losses': ['binary_crossentropy'],
        'activation':['relu', 'elu'],
        'last_activation': ['sigmoid']
    }

    return talos.Scan(
        x=x_data,
        y=y_data,
        model=concise_model,
        params=params,
        experiment_name="concise_analysis",
        round_limit=10)

def simple_analysis(x_data, y_data):
    params = {
        'first_neuron': [12, 24, 48],
        'activation': ['relu', 'elu'],
        'batch_size': [10, 20, 30]
    }

    return talos.Scan(
        x=x_data,
        y=y_data,
        params=params,
        model=simple_model,
        experiment_name="simple_analysis"
    )

def main():
    """The main function"""

    (train_x, train_y) = create_dataset("../dataset/output/*.json")
    (val_x, val_y) = create_dataset("../dataset/output/testset/*.json")

    # parameter_optimization(train_x, train_y)
    parameter_optimization(val_x, val_y)

if __name__ == "__main__":
    main()

