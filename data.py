import re,sys,random
import numpy as np
import pandas as pd

label_splitter = re.compile(r"[\d-]+")
feature_splitter = re.compile(r"[^,]")

class Dataset:

    def __init__(self,x,y):
        self.features = x
        self.label = y
        self.data = [x,y]

    def from_txt(file:str):
        """Return a Dataset from a text file.
        
        Args:
            file: filename
        """
        l = []
        
        with open(file, 'r') as f:
            file_as_list = f.read().splitlines()
            for i in file_as_list:
                label = int(label_splitter.findall(i)[0])
                features = label_splitter.findall(i)[1:]
                print(label_splitter.findall(i))
                print(features)
                features = [int(j) for j in features]

                l.append((features,label))
        
        return l



