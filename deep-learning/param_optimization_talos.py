"""json module for parsing json"""
import json
import random
import glob

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, Nadam, RMSprop
from keras.activations import softmax, relu, elu, sigmoid
from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy
from keras import backend as K


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
    model.add(Dense(params['first_neuron'], input_dim=90,
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.add(Dropout(params['dropout']))

    model.add(Dense(2, activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'],
                  metrics=['acc',f1_m,precision_m, recall_m]
                  # metrics=["accuracy"]
                  )
    
    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        callbacks=[talos.utils.live()],
                        epochs=params['epochs'],
                        verbose=0)

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
        Dense(1, activation=params['last_activation']))

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



def field_report_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Dense(10, input_dim=x_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer='normal'))
    
    model.add(Dropout(params['dropout']))
    
    # hidden_layers(model, params, 1)
    model.add(Dense(1, activation=params['last_activation'],
                    kernel_initializer='normal'))

    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  # metrics=['acc', fmeasure]
                  metrics=['acc',f1_m,precision_m, recall_m]
                  )
    
    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)
    
    return history, model

def parameter_optimization(x_data, y_data):
    """Find the optimal parameters for DL training"""

    # scan_object = simple_analysis(x_data, y_data)
    # scan_object = concise_analysis(x_data, y_data)
    # scan_object = comprehensive_analysis(x_data, y_data)
    scan_object = field_report_analysis(x_data, y_data)

    evaluate_object = talos.Evaluate(scan_object)
    evaluate_object.evaluate(x_data, y_data, folds=10, metric='val_acc', task='multi_label')

    return scan_object


def field_report_analysis(x_data, y_data):
    params = {
        'lr': (0.5, 5, 10),
        'first_neuron':[4, 8, 16, 32, 64],
        'hidden_layers':[0, 1, 2],
        'batch_size': (2, 30, 10),
        'epochs': [150],
        'dropout': (0, 0.5, 5),
        'weight_regulizer':[None],
        'emb_output_dims': [None],
        'shape':['brick','long_funnel'],
        'optimizer': [Adam, Nadam, RMSprop],
        'losses': [logcosh, binary_crossentropy],
        'activation':[relu, elu],
        'last_activation': [sigmoid]
    }

    return talos.Scan(
        x=x_data,
        y=y_data,
        params=params,
        model=field_report_model,
        experiment_name="field_report",
        fraction_limit=.001
    )

def comprehensive_analysis(x_data, y_data):
    params = {
        'lr': (0.1, 10, 10),
        'first_neuron':[4, 8, 16, 32, 64, 128],
        'batch_size': [2, 10, 20, 30, 40, 50],
        'epochs': [200],
        'dropout': (0, 0.40, 10),
        'optimizer': [Adam, Nadam],
        'loss': ['binary_crossentropy'],
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
    # then we can go ahead and set the parameter space
    params = {
        'first_neuron':[9,10,11],
        'hidden_layers':[0, 1, 2],
        'batch_size': [30],
        'epochs': [100],
        'dropout': [0],
        'kernel_initializer': ['uniform','normal'],
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

