"""json module for parsing json"""
import json
import random
import glob

# import tensorflow as tf
# import tensorflow.keras as keras
import numpy as np

# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
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

def create_model(optimizer="adam"):
    """Creates a keras model"""

    model = Sequential()
    # model.add(keras.layers.Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def parameter_optimization(x_data, y_data):
    """Find the optimal parameters for DL training"""

    # model = KerasClassifier(build_fn=create_model, verbose=0)
    # param_grid = get_param_grid("epochs")

    model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
    param_grid = get_param_grid("optimizer")

    if param_grid is None:
        print("No param grid")
        return

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
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

    if param == "epochs":
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50, 100]
        return dict(batch_size=batch_size, epochs=epochs)

    if param == "optimizer":
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        return dict(optimizer=optimizer)

    return None


def train_model(train_x, train_y, val_x, val_y):
    """Initializes the training of the model

    Also evaluates the training after run
    """

    model = create_model()
    model.fit(train_x, train_y, epochs=20)
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

    (train_x, train_y) = create_dataset("../dataset/output/*.json")
    (val_x, val_y) = create_dataset("../dataset/output/testset/*.json")

    # train_model(train_x, train_y, val_x, val_y)
    # parameter_optimization(train_x, train_y)
    parameter_optimization(val_x, val_y)

if __name__ == "__main__":
    main()
