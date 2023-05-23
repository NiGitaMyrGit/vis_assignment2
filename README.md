# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
The portfolio exam for Visual Analytics S22 consists of 4 projects; three class assignments and one self-assigned project. This is the repository for the second assignment in the portfolio.
## 1. Contributions
The code was produced by me, but with a lot of problem-solving looking into various Stack Overflow blog posts.

## 2. Initial assignment description by instructor
For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset.

You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier **and** one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn```.

## 2.1 Tips

- You should structure your project by having scripts saved in a folder called ```src```, and have a folder called ```out``` where you save the classification reports.
- Consider using some of the things we've seen in class, such as virtual environments and setup scripts.

## 2.2 Purpose

- To ensure that you can use ```scikit-learn``` to build simple benchmark classifiers on image classification data
- To demonstrate that you can build reproducible pipelines for machine learning projects
- To make sure that you can structure repos appropriately

## 3. Methods
**Logistic Regression model** The logireg_cifar10.py script uses different functions to load and preprocess the CIFAR_10 data set. The preprocessed data is then used as input for training a logistic regression model and predicting classes of the validation data based on the model which is saved in a classification report.
**Neural Network model** The neural_cifar10.py script uses the same loading and processing functions as mentioned in the section above. The output  are then used to train a network classification model. The model is compiled and trained on the input data, and an evaluation of the model is saved in a classification report.

# 4. Usage
This script was made using python 3.10.7, make sure this is your python version you run the script in. 
### 4.1 Installing packages
From the command line:
Clone this repository to your console by running the command `git clone https://github.com/NiGitaMyrGit/vis_assignment1.git`. This will copy the repository to the location you are currently in.
Then make sure you are located in the main folder, location can be changed by using the command `cd path\to\vis_assignment1`'. From here run the command `bash setup.sh` which will install all the required packages in order to run the script.

### 4.2 Dataset
This scritp uses the ```Cifar10``` dataset which is directly loaded into the script.

### 4.3 running the scripts
In the command line make sure you are located in the main folder.
Run the command `python3 src/logireg_cifar10.py` for running the ligistic regression script and `python3 src/neural_cifar10.py`
For this asignment I have used argparse so the user can customise the output path of the classification reports with the flag -r.
Eg.
`python3 logireg_cifar10.py -r custom/path/to/report.txt` or `python3 neural_cifar10.py -r custom/path/to/report.txt`
The default path is default='out/neuralnetwork_report.txt' which means the classification
## 5. Results - discussion
The accuracy of the logistic regression is 0.31 and 0.35 for the neural network. This is not overwhelmingly good results, 
 
