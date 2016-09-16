from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sys, os, time, numpy, glob, scipy, shutil
import argparse
import matplotlib.pyplot as plt
import matplotlib
import itertools
import operator
import datetime
from nltk import stem
from nltk.tokenize import word_tokenize
from pyAudioAnalysis import audioTrainTest
import cPickle
import random

MAX_FILES_PER_CLASS = 300

def classifierWrapper(classifier, testSample):
    R = classifier.predict(testSample.reshape(1,-1))[0]
    P = classifier.predict_proba(testSample.reshape(1,-1))[0]
    return [R, P]

def loadModel(modelName, isRegression=False):
    try:
        fo = open(modelName+"MEANS", "rb")
    except IOError:
            print "Load Model: Didn't find file"
            return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        if not isRegression:
            classNames = cPickle.load(fo)
    except:
        fo.close()
    fo.close()
    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)
    with open(modelName, 'rb') as fid:
        Classifier = cPickle.load(fid)    
    if isRegression:
        return(Classifier, MEAN, STD)
    else:
        return(Classifier, MEAN, STD, classNames)


def get_immediate_subdirectories(a_dir):    
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real time audio analysis")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks", dest="task", metavar="")

    trainFromDirs = tasks.add_parser("trainFromDirs", help="Train text classifier from list of directories (each directory corresponds to a different text class and has a set of documents)")
    trainFromDirs.add_argument("-i", "--input", required=True, help="Input directory where the the subdirectories-classes are stored")
    trainFromDirs.add_argument("--method",  choices=["svm", "knn", "randomforest","gradientboosting", "extratrees"], default="svm", help="Classification method")
    trainFromDirs.add_argument("--methodname", required=True,  help="Classifier path")

    classifyFile = tasks.add_parser("classifyFile", help="Classify an unknown document stored in a folder")
    classifyFile.add_argument("-i", "--input", required=True, help="Input file where the the unknown document is stored")
    #classifyFile.add_argument("--method",  choices=["svm", "knn", "randomforest","gradientboosting", "extratrees"], default="m1", help="Classification method")
    classifyFile.add_argument("--methodname", required=True,  help="Classifier folder path")

    return parser.parse_args()

def getListOfFilesInDir(dirName, pattern):    
    if os.path.isdir(dirName):
        strFilePattern = os.path.join(dirName, pattern)
    else:
        strFilePattern = dirName + pattern    
    textFilesList = []
    textFilesList.extend(glob.glob(strFilePattern))
    textFilesList = sorted(textFilesList)
    return textFilesList


def loadDictionaries(dictFolder):
    dictFiles = getListOfFilesInDir(dictFolder, "*.dict")
    porter = stem.porter.PorterStemmer()
    dicts = []
    for d in dictFiles:
        with open(d) as f:
            temp = f.readlines()
            temp = [(x.lower().replace("\n","").replace("\r","")) for x in temp]
            dicts.append(temp)
    return dicts         

def getFeaturesFromText(text, dicts):
    nDicts = len(dicts)
    curF = numpy.zeros((nDicts, 1))    
    words = word_tokenize(text.decode('utf-8'))
    words = [w.lower() for w in words]    
    for w in words:                                    
        for i, di in enumerate(dicts):
            if w in di:
                curF[i] += 1        
    curF /= len(text)
    return curF

def trainTextClassifiers(directoryPath, classifierType, classifierName):
    subdirectories = get_immediate_subdirectories(directoryPath)
    #tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features = 10000, stop_words='english')
    dicts = loadDictionaries("myDicts/")
    classNames = []    
    Features = []    
    # extract features from corpus
    for si, s in enumerate(subdirectories):                                         # for each directory in training data        
        print "Training folder {0:d} of {1:d} ({2:s})".format(si+1, len(subdirectories), s),
        files = getListOfFilesInDir(directoryPath + os.sep + s, "*")                # get list of files in directory
        if MAX_FILES_PER_CLASS > 0 and MAX_FILES_PER_CLASS < len(files):
            files = random.sample(files, MAX_FILES_PER_CLASS)
        print " - {0:d} files".format(len(files))
        classNames.append(s)
        for ifile, fi in enumerate(files):                                          # for each file in current class:
            with open(fi) as f:
                content = f.read() 
                curF = getFeaturesFromText(content, dicts)                           # get feature vector
            if ifile ==0 :                                                           # update feature matrix
                Features.append(curF.T)
            else:
                Features[-1] = numpy.concatenate((Features[-1], curF.T), axis = 0)

    # define classifier parameters
    if classifierType == "svm":
        classifierParams = numpy.array([0.001, 0.01,  0.5, 1.0, 5.0, 10.0])
    elif classifierType == "randomforest":
        classifierParams = numpy.array([10, 25, 50, 100,200,500])
    elif classifierType == "knn":
        classifierParams = numpy.array([1, 3, 5, 7, 9, 11, 13, 15])        
    elif classifierType == "gradientboosting":
        classifierParams = numpy.array([10, 25, 50, 100,200,500])        
    elif classifierType == "extratrees":
        classifierParams = numpy.array([10, 25, 50, 100,200,500])        

    # evaluate classifier and select best param
    nExp = 10
    bestParam  = audioTrainTest.evaluateClassifier(Features, subdirectories, nExp, classifierType, classifierParams, 0, 0.9)
                 
    # normalize features
    C = len(classNames)
    [featuresNorm, MEAN, STD] = audioTrainTest.normalizeFeatures(Features) 
    MEAN = MEAN.tolist(); STD = STD.tolist()
    featuresNew = featuresNorm

    # save the classifier to file
    if classifierType == "svm":
        Classifier = audioTrainTest.trainSVM(featuresNew, bestParam)
    elif classifierType == "randomforest":
        Classifier = audioTrainTest.trainRandomForest(featuresNew, bestParam)
    elif classifierType == "gradientboosting":
        Classifier = audioTrainTest.trainGradientBoosting(featuresNew, bestParam)
    elif classifierType == "extratrees":
        Classifier = audioTrainTest.trainExtraTrees(featuresNew, bestParam)

    if 'Classifier' in locals():
        with open(classifierName, 'wb') as fid:                                            # save to file
            cPickle.dump(Classifier, fid)            
        fo = open(classifierName + "MEANS", "wb")
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()                


def classifyFile(documentPath, modelPath):
    Threshold = 1.4
    t1 = time.time()
    Classifier, MEAN, STD, classNames = loadModel(modelPath)                               # load classifier
    t2 = time.time()
    with open(documentPath) as f:                                                          # load text fule
        text = f.read()
    t3 = time.time()        
    dicts = loadDictionaries("myDicts/")                                                   # load dicts
    t4 = time.time()
    F = getFeaturesFromText(text, dicts)                                                   # extract features        
    t5 = time.time()
    F = (F.flatten()- MEAN) / STD                                                          # normalize
    R, P = classifierWrapper(Classifier, F)
    t6 = time.time()
    '''
    print t2-t1
    print t3-t2
    print t4-t3
    print t5-t4
    print t6-t5
    for i in range(len(P)):
        print classNames[i], P[i]
    '''        

    meanP = 1.0 / float(len(classNames))

    Results = [(y,x) for (y,x) in sorted(zip(P, classNames), reverse=True)]
    for r in Results:
        if r[0] > Threshold * meanP:
            print r[1], r[0]
    
    return 0

if __name__ == "__main__":
    args = parse_arguments()
    if args.task == "trainFromDirs":
        trainTextClassifiers(args.input, args.method, args.methodname)
    if args.task == "classifyFile":
        classifyFile(args.input, args.methodname)        
