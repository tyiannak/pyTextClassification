
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
from collections import Counter
from nltk.corpus import stopwords
from operator import itemgetter

stop = set(stopwords.words('english'))

MAX_FILES_PER_CLASS = 50000

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real time audio analysis")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks", dest="task", metavar="")

    getFreqWordsFromDir = tasks.add_parser("getFreqWordsFromDir", help="Get most frequent words in a dir")
    getFreqWordsFromDir.add_argument("-i", "--input", required=True, help="Input directory")        
    
    return parser.parse_args()

def get_immediate_subdirectories(a_dir):    
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def getListOfFilesInDir(dirName, pattern):    
    if os.path.isdir(dirName):
        strFilePattern = os.path.join(dirName, pattern)
    else:
        strFilePattern = dirName + pattern    
    textFilesList = []
    textFilesList.extend(glob.glob(strFilePattern))
    textFilesList = sorted(textFilesList)
    return textFilesList

def getFreqWords(directoryPath):    
    files = getListOfFilesInDir(directoryPath, "*")                # get list of files in directory
    allWords = []
    count = 0
    if MAX_FILES_PER_CLASS > 0 and MAX_FILES_PER_CLASS < len(files):
        files = random.sample(files, MAX_FILES_PER_CLASS)        
    for ifile, fi in enumerate(files):                                          # for each file in current class:
        with open(fi) as f:
            content = f.read() 
            words = word_tokenize(content.decode('utf-8'))
            words = [w.lower() for w in words if w.lower() not in stop]                    
            words = list(set(words))
            allWords += words                
            count += 1
    #print allWords
    C = Counter(allWords)
    C = sorted(C.items(), key=itemgetter(1),reverse=True)        
    for c in C:
        if c[1] > 0.05 * float(count):
            print c[0], c[1] / float(count)

if __name__ == "__main__":
    
    # usage example: python utility_getFreqWords.py getFreqWordsFromDir -i moviePlots/Drama/

    args = parse_arguments()
    if args.task == "getFreqWordsFromDir":
        getFreqWords(args.input)
