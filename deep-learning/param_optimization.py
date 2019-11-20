"""json module for parsing json"""
import json
import random
import glob

import numpy as np

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier

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

def create_model(
        optimizer="Adam",
        activation="relu",
        init_mode="normal",
        learn_rate=0,
        momentum=0,
        neurons=100,
        weight_constraint=1,
        dropout_rate=0.2
        ):

    """Creates a keras model"""

    model = Sequential()
    # model.add(Flatten())
    model.add(
        Dense(
            neurons,
            kernel_initializer=init_mode,
            activation=activation,
            kernel_constraint=maxnorm(weight_constraint)
        )
    )

    model.add(Dropout(dropout_rate))

    model.add(Dense(2, activation='softmax'))

    if learn_rate > 0.0 and momentum > 0:
        optimizer = SGD(lr=learn_rate, momentum=momentum)


    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def parameter_optimization(x_data, y_data):
    """Find the optimal parameters for DL training"""

    # model = KerasClassifier(build_fn=create_model, verbose=0)
    # param_grid = get_param_grid("epochs_and_batch_size")

    model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
    # param_grid = get_param_grid("optimization_method")
    # param_grid = get_param_grid("SGD")
    # param_grid = get_param_grid("weight_initialization")
    # param_grid = get_param_grid("activation_function")
    param_grid = get_param_grid("num_of_neurons")
    # param_grid = get_param_grid("dropout_rate")

    if param_grid is None:
        print("No param grid")
        return

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=2, cv=3)
    grid_result = grid.fit(x_data, y_data, verbose=0)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def get_param_grid(param):
    """Returns param grids based on values for the parameter optimization"""
    ret = None

    if param == "epochs_and_batch_size":
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50, 100, 150]
        ret = dict(batch_size=batch_size, epochs=epochs)

    if param == "optimization_method":
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        ret = dict(optimizer=optimizer)

    if param == "SGD":
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        ret = dict(learn_rate=learn_rate, momentum=momentum)

    if param == "weight_initialization":
        init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero',
                     'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
        ret = dict(init_mode=init_mode)

    if param == "activation_function":
        activation = ['softmax', 'softplus', 'softsign', 'relu',
                      'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        ret = dict(activation=activation)

    if param == "dropout_rate":
        weight_constraint = [1, 2, 3, 4, 5]
        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ret = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)

    if param == "num_of_neurons":
        neurons = [1, 5, 10, 15, 20, 25, 30, 50, 100, 150]
        ret = dict(neurons=neurons)

    return ret


def main():
    """The main function"""

    (train_x, train_y) = create_dataset("../dataset/output/*.json")
    (val_x, val_y) = create_dataset("../dataset/output/testset/*.json")

    # parameter_optimization(train_x, train_y)
    parameter_optimization(val_x, val_y)

if __name__ == "__main__":
    main()
