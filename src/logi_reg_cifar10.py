# path tools
import os
import sys
sys.path.append(".")
import joblib
import pandas as pd
import numpy as np
import cv2
# importing Ross' classifier utils
import utils.classifier_utils as clf
# data loader
from tensorflow.keras.datasets import cifar10

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# importing argument parser
import argparse
from argparse import ArgumentParser
parser = ArgumentParser()

parser = argparse.ArgumentParser(description='logistic regression model')

# load and preprocess data
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # making labels, since dataset does not contain premade labels
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
def train_model(y_train,X_test_dataset, y_test,):
    clf = LogisticRegression(penalty="none",
                            tol=0.1,
                            verbose=True,
                            solver="saga",
                            multi_class="multinomial").fit(X_train_dataset, y_train)
    y_pred = clf.predict(X_test_dataset)
    return y_pred, clf
# save classification report
def classifier_report(y_test, y_pred):
    report = metrics.classification_report(y_test, y_pred, target_names=labels)
    return report

# main function
def main():
    #load data
    X_train_dataset, X_test_dataset, y_train, y_test = load_data()
    #Train model
    y_pred, clf = train_model(y_train, X_test_dataset, y_test)

    #Save classification report
    report = metrics.classifier_report(y_test, y_pred)
    with open('classification_report.txt', 'w') as f:
        f.write(classifier_metrics)
    # save the trained models and vectorizers
    from joblib import dump, load
    # save the model
    dump(classifier, os.path.join("out", "logi_reg_model.joblib"))

# calling main function
if __name__== "__main__":
    main()