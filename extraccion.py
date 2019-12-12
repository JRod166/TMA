import logging
import os

import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display

import chroma

import sys

np.set_printoptions(threshold=sys.maxsize)

#Globals
FFT_SIZE=2048
HOP_SIZE=512
SAMPLING_RATE = 44100

def chromaFeatures(file):
    try:
        logging.info("Loading file {}".format(os.path.basename(file)))
        audio, sr = librosa.load(format(file),sr=SAMPLING_RATE)
    except Exception as e:
        logging.info("Couldn't load file {}".format(os.path.basename(file)))
        logging.info("Error: {}".format(e))
        return

    try:
        logging.info("Short Time Fourier Transformation")
        D=librosa.stft(audio)
    except Exception as e:
        logging.info("Couldn't perform STFT")
        logging.info("Error: {}".format(e))
        return

    try:
        logging.info("HPSS based on median")
        D_harmonic, D_percussive = librosa.decompose.hpss(D)
        y_harm=librosa.istft(D_harmonic)
        # y_harm = librosa.effects.harmonic(y=audio)
    except Exception as e:
        logging,info("Couldn't perform HPSS")
        logging.info("Error: {}".format(e))
        return

    features = {}
    try:
        logging.info("Extraction of chroma features")
        chroma_os_harm=librosa.feature.chroma_cqt(y=y_harm,sr=sr,
                                    hop_length=HOP_SIZE, threshold=0.005)
        chroma_filter = np.minimum(chroma_os_harm,
                           librosa.decompose.nn_filter(chroma_os_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))
        features["sequenceChroma"] = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
    except Exception as e:
        logging.info("Couldn't perform chroma features extraction")
        logging.info("Error: {}".format(e))
        return


    plt.figure(figsize=(12, 18))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(features["sequenceChroma"], y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.ylabel('3x-over')
    plt.tight_layout()
    plt.savefig("{}.png".format(file), box_inches='tight')

    return features["sequenceChroma"].T
