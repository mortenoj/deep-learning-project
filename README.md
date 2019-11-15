# deep-learning-project

## csgo-ui-cv

A python project which uses OpenCV to read equipment from an image of a CSGO match.

## dataset

A Golang project which creates the dataset to be used to train the deeplearning neural network.

Firtsly, it webscrapes (HLTV.org)[https://hltv.org] for demo matches

Then it uses (demoinfocs-golang)[https://github.com/markus-wa/demoinfocs-golang]
 to extract the equipment of both teams at every round freeze-time-end along with which team won that round.
 
Lastly, it stores this data in a format which can be used to train a deep neural network.


