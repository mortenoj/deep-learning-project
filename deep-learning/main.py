import json
import random
import glob

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


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

    return (data_x, data_y)

def create_model():
    """Creates a keras model"""

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid))
    # model.add(tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid))
    # model.add(tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  # loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    """The main function"""

    (train_x, train_y) = create_dataset("../dataset/output/*.json")
    (test_x, test_y) = create_dataset("../dataset/output/testset/*.json")

    model = create_model()
    model.fit(train_x, train_y, epochs=20)

    print("\nEvaluating...\n")

    (val_loss, val_acc) = model.evaluate(test_x, test_y)

    print("evaluated loss: ", val_loss)
    print("evaluated accuracy: ", val_acc)
    print("\n\n")

    correct = 0
    for i, pred in enumerate(model.predict(test_x)):
        if np.argmax(pred) == test_y[i]:
            correct += 1

    print("Actual accurancy: ", correct / len(test_x))


if __name__ == "__main__":
    main()
