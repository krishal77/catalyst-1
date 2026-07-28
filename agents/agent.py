import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if 'SUMO_HOME' in os.environ:
    tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools_path)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

sumoCfg = [
    'sumo-gui',
    '-c', 'RL.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000'
]

traci.start(sumoCfg)

carsEB1 = 0
carsEB2 = 0
carsEB3 = 0
carsSB1 = 0
carsSB2 = 0
carsSB3 = 0
phaseNow = 0

maxSteps = 10000

learnRate = 0.1

discount = 0.9

randChance = 0.1

myActions = [0, 1]

minGreenSteps = 100
lastSwitchStep = -minGreenSteps


def makeModel(numInputs, numOutputs):
    myModel = keras.Sequential()
    myModel.add(layers.Input(shape=(numInputs,)))
    myModel.add(layers.Dense(24, activation='relu'))
    myModel.add(layers.Dense(24, activation='relu'))
    myModel.add(layers.Dense(numOutputs, activation='linear'))
    myModel.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    return myModel


def makeArray(stuff):
    return np.array(stuff, dtype=np.float32).reshape((1, -1))


numInputs = 7
numOutputs = len(myActions)
myModel = makeModel(numInputs, numOutputs)


def getBestQ(s):
    arr1 = makeArray(s)
    qStuff = myModel.predict(arr1, verbose=0)[0]
    return np.max(qStuff)


def calcReward(s):
    totalCars = sum(s[:-1])
    rewardVal = -float(totalCars)
    return rewardVal


def getStuffNow():
    global carsEB1, carsEB2, carsEB3, carsSB1, carsSB2, carsSB3, phaseNow

    det1 = "Node1_2_EB_0"
    det2 = "Node1_2_EB_1"
    det3 = "Node1_2_EB_2"

    det4 = "Node2_7_SB_0"
    det5 = "Node2_7_SB_1"
    det6 = "Node2_7_SB_2"

    lightId = "Node2"

    carsEB1 = getCarsWaiting(det1)
    carsEB2 = getCarsWaiting(det2)
    carsEB3 = getCarsWaiting(det3)

    carsSB1 = getCarsWaiting(det4)
    carsSB2 = getCarsWaiting(det5)
    carsSB3 = getCarsWaiting(det6)

    phaseNow = getPhaseNow(lightId)

    return (carsEB1, carsEB2, carsEB3, carsSB1, carsSB2, carsSB3, phaseNow)


def doAction(actionNum, lightId2="Node5"):
    global lastSwitchStep

    if actionNum == 0:
        return
    elif actionNum == 1:
        if stepNow - lastSwitchStep >= minGreenSteps:
            prog1 = traci.trafficlight.getAllProgramLogics(lightId2)[0]
            howManyPhases = len(prog1.phases)
            nextPhase = (getPhaseNow(lightId2) + 1) % howManyPhases
            traci.trafficlight.setPhase(lightId2, nextPhase)
            lastSwitchStep = stepNow


def fixQTable(oldS, actionNum, rewardVal, newS):
    oldArr = makeArray(oldS)
    oldQs = myModel.predict(oldArr, verbose=0)[0]

    newArr = makeArray(newS)
    newQs = myModel.predict(newArr, verbose=0)[0]
    bestNextQ = np.max(newQs)

    oldQs[actionNum] = oldQs[actionNum] + learnRate * (rewardVal + discount * bestNextQ - oldQs[actionNum])

    myModel.fit(oldArr, np.array([oldQs]), verbose=0)


def pickAction(s):
    if random.random() < randChance:
        return random.choice(myActions)
    else:
        arr2 = makeArray(s)
        qVals2 = myModel.predict(arr2, verbose=0)[0]
        return int(np.argmax(qVals2))


def getCarsWaiting(detId):
    return traci.lanearea.getLastStepVehicleNumber(detId)


def getPhaseNow(lightId3):
    return traci.trafficlight.getPhase(lightId3)


stepList = []
rewardList = []
carsList = []

totalReward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
for stepNow in range(maxSteps):

    stateNow = getStuffNow()
    actionNow = pickAction(stateNow)
    doAction(actionNow)

    traci.simulationStep()

    stateNext = getStuffNow()
    rewardNow = calcReward(stateNext)
    totalReward += rewardNow

    fixQTable(stateNow, actionNow, rewardNow, stateNext)

    qValsNow = myModel.predict(makeArray(stateNow), verbose=0)[0]

    if stepNow % 1 == 0:
        qValsNow = myModel.predict(makeArray(stateNow), verbose=0)[0]
        print(f"Step {stepNow}, Current_State: {stateNow}, Action: {actionNow}, New_State: {stateNext}, Reward: {rewardNow:.2f}, Cumulative Reward: {totalReward:.2f}, Q-values(current_state): {qValsNow}")
        stepList.append(stepNow)
        rewardList.append(totalReward)
        carsList.append(sum(stateNext[:-1]))


traci.close()

print("model summary:")
myModel.summary()

