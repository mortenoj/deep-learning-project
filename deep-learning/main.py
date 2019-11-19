import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import json
import random
import glob


def create_dataset(path):
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
    return dataset


training_data = create_dataset("../dataset/output/*.json")
test_set = create_dataset("../dataset/output/testset/*.json")

random.shuffle(training_data)

train_x = []
train_y = []
for features,label in training_data:
    train_x.append(features)
    train_y.append(label)

test_x = []
test_y = []
for features,label in test_set:
    test_x.append(features)
    test_y.append(label)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10)

print("\nEvaluating...\n")
val_loss, val_acc = model.evaluate(test_x, test_y)
print("evaluated loss: ", val_loss)
print("evaluated accuracy: ", val_acc)

print("\n\n")

predictions = model.predict(test_x)
print(np.argmax(predictions[45]))
print(test_y[45])





