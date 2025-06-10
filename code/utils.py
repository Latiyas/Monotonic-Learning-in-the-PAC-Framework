import os, math, random, time
import numpy as np
import pickle


### load .pkl file
def load_file(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


### save .pkl file
def save_file(dataset, path):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)


### look for .pkl files that begins with XX
def find_files(path, name):
    files = os.listdir(path)
    targets = []
    for f in files:
        if name in f and f.endswith('.pkl'):
            targets.append(f)
    return targets