import argparse
import extraccion
import ruido
import reglas
import tomidi
import logging
import os

import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display

import chroma

import sys



if __name__ == '__main__':
    #Setup logger
    parser = argparse.ArgumentParser(description="Proyecto de tesis",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_file",
                        action="store",
                        help="Input audio file")
    parser.add_argument("-D",
                        action="store_const",
                        dest="Debug",
                        const=True,
                        default=False,
                        help="Debug mode")
    args=parser.parse_args()
    if(args.Debug):
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)

    plt.figure(figsize=(12, 18))

    chromaFeatures=extraccion.chromaFeatures(args.audio_file)

    chromaFeatures=ruido.supresion(chromaFeatures)

    chromaFeatures=reglas.rules(chromaFeatures)

    plt.savefig("{}.png".format(args.audio_file), box_inches='tight')
    print (chromaFeatures)

    tomidi.chroma_to_midi(chromaFeatures,args.audio_file+".mid")
