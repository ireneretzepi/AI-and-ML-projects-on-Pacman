# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree


# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray

    # def CrossValidation(self):

    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])


        # data and target are both arrays of arbitrary length.
        # data is an array of arrays of integers (0 or 1) indicating state.
        # target is an array of integers 0-3 indicating the action taken in that state.

        # *********************************************
        #
        # Running NAIVE BAYES classifier

        # Returns the number of nested lists (number of samples)
        num_samples = len(self.data[:][:])
        # Returns the number of features in the training data
        num_features = len(self.data[0][:])


        # building Naive Bayes classifier with the training data
        self.probability_dict, self.prior, self.y_count = self.NaiveBayesTraining(self.data, self.target)


    # *********************************************
    #     Training the Naive Bays classifier
    # *********************************************

    def NaiveBayesTraining(self, x_train, y_train):

        # Calculates the number of times each class appears in the training set
        y_count = [0.0, 0.0, 0.0, 0.0]
        for i in range(len(y_train)):
            if y_train[i] == 0:
                y_count[0] += 1
            if y_train[i] == 1:
                y_count[1] += 1
            if y_train[i] == 2:
                y_count[2] += 1
            if y_train[i] == 3:
                y_count[3] += 1
            if y_train[i] == 4:
                y_count[4] += 1

        # Returns the total number of entries in the dataset
        total = sum(y_count)

        # Calculates the prior value for each class
        prior = [y / total for y in y_count]

        # Calculates the number of times each feature appears in each class
        num_features = len(x_train[0][:])
        probability_dict = {0: [0] * num_features, 1: [0] * num_features, 2: [0] * num_features, 3: [0] * num_features}

        index = 0
        for i in y_train:
            if i == 0:
                probability_dict[0] += np.array(x_train[index])
            if i == 1:
                probability_dict[1] += np.array(x_train[index])
            if i == 2:
                probability_dict[2] += np.array(x_train[index])
            if i == 3:
                probability_dict[3] += np.array(x_train[index])
            index += 1


        # Normalizes by the number of times each class appears in the data to get probabilities
        keys = list(probability_dict.keys())
        for i in range(len(probability_dict)):
            probability_dict[keys[i]] = np.divide(probability_dict[keys[i]], y_count[i])


        return probability_dict, prior, y_count


    # Uses the trained Naive Bayes classifier to make predictions
    def NaiveBayesTest(self, features):

        num_features = len(features)
        features_prob = [1, 1, 1, 1]

        # Calculates the probability of each feature belonging at each class
        # using prior and conditional probability
        for i in range(len(self.prior)):
            sum = 0
            for j in range(num_features):
                sum += self.probability_dict[i][j] * features[j]
            features_prob[i] = features_prob[i] * self.prior[i] * sum
        print features_prob

        # Returns the maximum value of the probabilities which indicates the best action to perform
        return features_prob.index(max(features_prob))



    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"

        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)

        # *****************************************************
        #
        # Here you should insert code to call the classifier to
        # decide what to do based on features and use it to decide
        # what action to take.
        #
        # *******************************************************

        # Classifies features using the Naive Bayes classifier created above
        # Returns the class with the highest probability
        classification = self.NaiveBayesTest(features)

        nb = MultinomialNB()
        nb = nb.fit(self.data, self.target)
        # bestPredictedAction = self.convertNumberToMove(nb)

        # Converts the class number from a string to an int
        bestPredictedAction = self.convertNumberToMove(classification)

        # Get the actions we can try.
        legal = api.legalActions(state)

        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        if bestPredictedAction in legal:
            return api.makeMove(bestPredictedAction, legal)
        else:
            return api.makeMove(random.choice(legal), legal)
