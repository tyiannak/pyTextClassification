# pyTextClassification
Training and using classifiers for textual documents

## General
pyTextClassification is a simple python library that can be used to train and use text classifiers. 
It can be trained using a corpus of text documents organized in folders, each folder corresponding to a different content class.

## Installation and dependencies

 * pip dependencies:
 ```
pip install numpy matplotlib scipy sklearn nltk
```

 * [pyAudioAnalysis] (https://github.com/tyiannak/pyAudioAnalysis) used for training and evaluating classifiers

## Train a classifier
In order to train a classifier based on a dataset, the following command must be used:
 ```
python textClassification.py trainFromDirs -i <datasetPath> --method <svm or knn or randomforest or gradientboosting or extratrees> --methodname <modelFileName>
```

`<datasetPath>` is the path of the training corpus. This path must contain a list of folders, each one corresponding to a different content class. Each folder contains a list of filenames (no extension assumed) which correspond to documents belonging to this class

`<modelFileName>` is the path where the extracted model is stored

Feature extraction is done using a set of predefined (static) dictionaries, stored in the `myDicts/` folder. For each dictionary, a separate feature value is extracted.

Example:
 ```
python textClassification.py trainFromDirs -i moviePlotsSmall/ --method svm --methodname svmMoviesPlot7Classes
```

## Apply a classifier
Given a trained model, and an unknown document, the following command syntax is used to classify the document:
 ```
python textClassification.py classifyFile -i <pathToUnknownDocument> --methodname <modelFileName>
```

This repository already contains a trained SVM model (`svmMoviesPlot7Classes`) that discriminates between 7 classes of movie plots. The files `sample_pulpFiction`, `sample_forestgump` and `sample_lordoftherings` contain two plot examples that can be used as unknown documents for testing. 


In order to classify these two files using `svmMoviesPlot7Classes`, the following command must be executed:
 ```
python textClassification.py classifyFile -i sample_pulpFiction --methodname svmMoviesPlot7Classes

python textClassification.py classifyFile -i sample_forestgump --methodname svmMoviesPlot7Classes

python textClassification.py classifyFile -i sample_lordoftherings --methodname svmMoviesPlot7Classes
```
