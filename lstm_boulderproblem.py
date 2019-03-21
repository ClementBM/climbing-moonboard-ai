# -*- coding: utf-8 -*-
"""LSTM BoulderProblem.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_Q9qKOtUSA5pi4jFysn1jnQNgjiPM6Lq
"""

# LSTM Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from pandas import read_csv
import json
from matplotlib import pyplot
import math
import os
from os import listdir
from os.path import isfile, join
import numpy as np

# Prepare data
abcIndex = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

gradeIndex = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+']


dataFiles = [f for f in listdir("sample_data") if isfile(join("sample_data", f))]

boulderProblemsMoves = []
bGrades = []
  
for dataFile in dataFiles:
  bProblemsPath = os.path.join("sample_data", dataFile)
  bGrade = dataFile.replace(".json","")
  bGradeIndex = gradeIndex.index(bGrade)
  gradeHotOne = [0] * len(gradeIndex)
  gradeHotOne[bGradeIndex] = 1
  
  with open(bProblemsPath) as jsonData:
      jsonRaw = jsonData.read()
      boulderProblems5 = json.loads(jsonRaw)

  exCount = len(boulderProblems5['Data'])

  eobVect = [0] * 199
  eobVect[198] = 1

  for i in range(0, exCount):
    moves = boulderProblems5['Data'][i]['Moves']
    moveCount = len(moves)
    boulderProblemMoves = []
    
    holdLocations = boulderProblems5['Data'][i]['Locations']
    holdCount = len(holdLocations)
    distances = []
    
    if(moveCount != holdCount):
      print("moveCount != holdCount")
    
    for j in range(0, moveCount):
      ## Get Hold Index
      hold = moves[j]['Description']
      letterIndex = abcIndex.index(hold[0])
      number = int(hold[1:])
      hotOneVect = [0] * 199 # 199 (distance)
      hotOneVect[((letterIndex * 18) + number - 1)] = 1
      boulderProblemMoves.append(hotOneVect)
      
      ## Get distance
      currentHoldLocation = holdLocations[j]
      if(j == 0):
        distances.append([0, 0, 0])
      else:
        oldHoldLocation = holdLocations[j-1]
        xDistance = currentHoldLocation['X'] - oldHoldLocation['X']
        yDistance = currentHoldLocation['Y'] - oldHoldLocation['Y']
        distance = math.sqrt(xDistance**2 + yDistance**2)
        distances.append([xDistance**2, yDistance**2, distance**2])
      ##
      
    # Add End Of Boulder vector
    boulderProblemMoves.append(eobVect)
    distances.append([0, 0, 0])
    
    # Only Distances
    # boulderProblemsMoves.append(distances)

    # Distances and Holds
    distancesFlat = [y for x in distances for y in x]
    distancesFlatMax = max(distancesFlat)
    #print(distancesFlatMax)
    distancesNorm = np.log(np.add(np.abs(distances), 1)) # np.divide(distances, 500)
    boulderProblemsMoves.append(np.concatenate((boulderProblemMoves, np.divide(distancesNorm, 5)), axis=1))

    # Only holds
    # boulderProblemsMoves.append(boulderProblemMoves)

    # y
    bGrades.append(bGradeIndex)

    # Shapes

# print(bGrades)
# print(boulderProblemsMoves)
print(len(boulderProblemsMoves))

max_length = max([len(s) for s in boulderProblemsMoves])
print(max_length)

X = np.array(boulderProblemsMoves)
Xpad = pad_sequences(X, maxlen=max_length, padding='post')

print(Xpad.shape)

y = np.array(bGrades)

# split into train and test sets
# Xpad = np.reshape(Xpad,(Xpad.shape[0],Xpad.shape[1],1))
RANDOM_SEED = 5
X_train, X_test, y_train, y_test = train_test_split(Xpad, y, test_size=0.2, random_state=RANDOM_SEED)

print("X_train")
print(X_train.shape)
print("X_test")
print(X_test.shape)
print(y_test)

print(X_train.shape[1])
print(X_train.shape[2])
# Model
model = Sequential()
# first hidden layer
model.add(LSTM(input_shape=(X_train.shape[1], X_train.shape[2]), units=30, return_sequences=True))
model.add(Dropout(0.2))
# second hidden layer
model.add(LSTM(20, return_sequences=False))
model.add(Dropout(0.2))
# 1 neuron in the output layer
model.add(Dense(10))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(X_train, 
                    y_train, 
                    epochs=50,
                    validation_data=(X_test, y_test),
                    batch_size=64,
                    verbose=2, 
                    shuffle=False)
model.summary()

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
predictions = model.predict(X_test)
meanAbsError = 0
maeDifferences = []

for i in range(0, len(predictions)):
  prediction = predictions[i][0]
  roundedPrediction = int(round(prediction))
  diff = roundedPrediction - y_test[i]
  meanAbsError += abs(diff)
  maeDifferences.append(abs(diff))
  print('Predict {} but was {} / {} round to {} expect {} / diff {}'.format(gradeIndex[roundedPrediction], gradeIndex[y_test[i]], prediction, roundedPrediction, y_test[i], diff))
  print("###")
  
print("MAE")
print(meanAbsError / len(predictions))

maeDifferences = np.array(maeDifferences)
hist, bins = np.histogram(maeDifferences)

pyplot.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]))
pyplot.plot()

print(maeDifferences)