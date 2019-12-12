import logging
import os

import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display

import chroma

import sys




mayor = [0,4,7]
menor = [0,3,7]
disminuida = [0,3,6]
aumentada = [0,4,8]
acordes = [mayor,menor,disminuida,aumentada]

treshold= 0.05

def apply(features,mode,pos):
    # print (mode)
    chord=[]
    for item in mode:
        if (features[(pos+item)%11]>treshold):
            chord.append(features[(pos+item)%11])
        # print (features[(pos+item)%11])
    return chord

def sum (chord):
    sum =0
    for item in chord:
        sum+=item
    return sum

def getBest(framePosibilities):
    max=0
    final_positions=[]
    for i in range (0,len(framePosibilities)):
        if(framePosibilities[i][2]>max):
            max=framePosibilities[i][2]
    for i in range (0,len(framePosibilities)):
        if(framePosibilities[i][2]>=max-treshold):
            for item in framePosibilities[i][0]:
            # final_positions.append(framePosibilities[i])
                # print (item+framePosibilities[i][1]%11)
                final_positions.append((item+framePosibilities[i][1])%11)
    return final_positions



def rules(chromaFeatures):
    framePosibilities=[]
    # for i in range (0,len(chromaFeatures)):
    logging.info("Parsing chord rules")
    for i in range (0,len(chromaFeatures)):
        if(i%100==0):
            logging.info("Rules: {} frames".format(i))
        for j in range (0,len(chromaFeatures[i])):
            if (chromaFeatures[i][j]!=0):
                for rule in acordes:
                    chord=apply(chromaFeatures[i],rule,j)
                    framePosibilities.append([rule,j,sum(chord)])
        # print(chromaFeatures[i])
        newFeatures=np.zeros(12)
        # print(newFeatures)
        for j in getBest(framePosibilities):
            newFeatures[j] = chromaFeatures[i][j]
        chromaFeatures[i]=newFeatures
        # print(chromaFeatures[i])
        # print(getBest(framePosibilities))
    plt.subplot(2, 1, 2)
    librosa.display.specshow(chromaFeatures.T, y_axis='chroma', x_axis='time')
    plt.colorbar()

    return chromaFeatures
