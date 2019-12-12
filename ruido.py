import logging
import os

import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display
import math

from sklearn.cluster import KMeans
import sys

threshold_values = {}
h = [1]

np.set_printoptions(threshold=sys.maxsize)

#Globals
FFT_SIZE=2048
HOP_SIZE=512
SAMPLING_RATE = 44100

def distance(first,second):
    avg_First=0
    avg_Second=0

    for i in range (0,len(first)):
        avg_First+=first[i]
    avg_First = avg_First/len(first)

    for i in range (0,len(second)):
        avg_Second+=second[i]
    avg_Second = avg_First/len(second)

    return abs(avg_First-avg_Second)

def maxArray(Array):
    first = len(Array[0])
    second = len(Array[1])
    third = len(Array[2])
    if (first >= second and first >= third):
        return 0
    if (second>= first and second >= third):
        return 1
    if (third >= first and third >= second):
        return 2


def clusterize(D,P):
    y=D
    if len(y) < 3:
        return
    x = range(len(y))
    m = np.matrix([x, y]).transpose()
    # cluster=KMeans(m,3)
    cluster=KMeans(n_clusters=3).fit(m)
    clusters=[[],[],[]]
    labels = cluster.labels_
    for i in range (0,len(labels)):
        clusters[labels[i]].append(D[i])
    max=maxArray(clusters)
    label = 3
    if (max==0):
        if (distance(clusters[0],clusters[1])>P):
            label=1
        if (distance(clusters[0],clusters[2])>P):
            label=2
    if (max==1):
        if (distance(clusters[0],clusters[1])>P):
            label=0
        if (distance(clusters[1],clusters[2])>P):
            label=2
    if (max==2):
        if (distance(clusters[0],clusters[2])>P):
            label=0
        if (distance(clusters[1],clusters[2])>P):
            label=1
    if (label == 3 ):
        return
    else :
        delete = []
        for i in range (0,len(labels)):
            if ( labels[i] == label ):
                delete.append(i)
    return delete




def supresion(chromaFeatures):
    logging.info("Analyzing Noise per frame")
    for i in range (0,len(chromaFeatures)):
        if(i%100==0):
            logging.info("Noise: {} frames".format(i))
        D=[]
        sum=0
        for j in range (0,len(chromaFeatures[i])):
            sum+=chromaFeatures[i][j]
        P = sum / len(chromaFeatures[i])
        for j in range (0,len(chromaFeatures[i])):
             if (chromaFeatures[i][j]!=0):
                 D.append(abs(chromaFeatures[i][j]-P))
        delete = clusterize(D,P)
        if delete:
            for item in delete:
                chromaFeatures[i][item]=0
    return chromaFeatures
