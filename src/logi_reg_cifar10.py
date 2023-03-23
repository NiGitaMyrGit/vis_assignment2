# path tools
import os

# data loader
from tensorflow.keras.datasets import cifar10

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classificatio models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

