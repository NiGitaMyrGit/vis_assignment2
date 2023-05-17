#!/usr/bin/env python3
# path tools
import os
import sys
import pandas as pd
import numpy as np
import cv2
# importing Ross' classifier utils
import utils.classifier_utils as clfimport

# data loader
from tensorflow.keras.datasets import cifar10

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classification models
from sklearn.linear_model import LogisticRegression
# importing argument parser
import argparse
from argparse import ArgumentParser
parser = ArgumentParser()

parser = argparse.ArgumentParser(description='logistic regression model')

# load and preprocess data
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # making labels, since dataset does not contain premade labels
    
    # convert data to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    # scaling
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0
    # reshape data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
    return X_train_dataset, X_test_dataset, y_train, y_test

# train classifier
def train_model(X_train_dataset, X_test_dataset, y_train):
    clf = LogisticRegression(penalty="none",
                            tol=0.1,
                            verbose=True,
                            solver="saga",
                            multi_class="multinomial").fit(X_train_dataset, y_train)
    y_pred = clf.predict(X_test_dataset)
    return y_pred, clf


# save classification report
def classifier_report(y_test, y_pred):
    labels = ["airplane",
              "automobile", 
              "bird", 
              "cat", 
              "deer", 
              "dog", 
              "frog", 
              "horse", 
              "ship", 
              "truck"]
    report = classification_report(y_test, y_pred, target_names=labels)
    return report

# main function
def main():
    #load data
    X_train_dataset, X_test_dataset, y_train, y_test = load_data()
    #Train model
    y_pred, clf = train_model(X_train_dataset, X_test_dataset, y_train, y_test)
    # save model
     #Saving classification report
    classifier_metrics = classifier_report(y_test, y_pred)
    with open(os.path.join("out","classification_report.txt"), 'w') as f:
        f.write(classifier_metrics)

# calling main function
if __name__== "__main__":
    main()