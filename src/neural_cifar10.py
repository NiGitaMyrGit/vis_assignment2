#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from argparse import ArgumentParser

from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

# Argument Parser
parser = ArgumentParser(description='Neural Network Classifier')
parser.add_argument('-r','--report_path', type=str, default='out/neuralnetwork_report.txt',
                    help='Path to save the classification report')
args = parser.parse_args()

# Load data
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    X_train_scaled = X_train_grey / 255.0
    X_test_scaled = X_test_grey / 255.0
    X_train_dataset = X_train_scaled.reshape((X_train_scaled.shape[0], -1))
    X_test_dataset = X_test_scaled.reshape((X_test_scaled.shape[0], -1))
    return X_train_dataset, X_test_dataset, y_train.flatten(), y_test.flatten()

# Train model
def train_model(X_train_dataset, X_test_dataset, y_train):
    clf = MLPClassifier(random_state=42,
                        hidden_layer_sizes=(64, 10),
                        learning_rate="adaptive",
                        early_stopping=True,
                        verbose=True,
                        max_iter=20).fit(X_train_dataset, y_train)
    y_pred = clf.predict(X_test_dataset)
    return y_pred

# Generate classification report
def classifier_report(y_pred, y_test):
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    report = classification_report(y_test, y_pred, target_names=labels)
    return report

# Main function
def main():
    # Load data
    X_train_dataset, X_test_dataset, y_train, y_test = load_data()

    # Train model
    y_pred = train_model(X_train_dataset, X_test_dataset, y_train)

    # Generate classification report
    classifier_metrics = classifier_report(y_pred, y_test)

    # Save classification report
    with open(args.report_path, 'w') as f:
        f.write(classifier_metrics)

# Call main function
if __name__ == "__main__":
    main()
