import os
from scipy.io.wavfile import read
import numpy as np
import python_speech_features as sp
import matplotlib.pyplot as plt

def setup():
    file = open('setup.txt', 'r')
    file_lenght = len(file.readlines())
    file.seek(0)
    frame_length = file.readline()
    frame_length = float(frame_length[18:len(frame_length) - 1:])
    mel_filters = file.readline()
    mel_filters = int(mel_filters[24:len(mel_filters) - 1:])
    predictor_order = file.readline()
    predictor_order = int(predictor_order[16:len(predictor_order) - 1:])
    cepstral_coefficient = file.readline()
    cepstral_coefficient = int(cepstral_coefficient[36:len(cepstral_coefficient) - 1:])
    GMM_components = file.readline()
    GMM_components = int(GMM_components[24:len(GMM_components) - 1:])
    covariance_type = file.readline()
    covariance_type = covariance_type[26:len(covariance_type) - 1:]
    permutations = file.readline()
    permutations = int(permutations[36:len(permutations):])
    return frame_length, mel_filters, predictor_order, cepstral_coefficient, GMM_components, covariance_type, permutations

def import_files(folder):
    Fs = []
    data = []
    for i in range(10):
        Fs.append([])
        data.append([])
        for file in os.listdir(folder):
            if file[5:8] == '_%d_' % i:
                rate, wavefile = read(folder + '/' + file)
                Fs[i].append(rate)
                data[i].append(wavefile)
    return Fs, data

def compute_mfcc(Fs, data, frame_length):
    mfcc_matrix = []
    for i in range(10):
        number_data = []
        for j in range(len(data[i][:])):
            mfcc_data = sp.base.mfcc(data[i][j], samplerate=Fs[i][j], winlen=frame_length, lowfreq=50, highfreq=8000, winstep=0.01, appendEnergy=True, nfft=882)
            number_data = np.concatenate((number_data, mfcc_data[:, 0]))
        mfcc_matrix.append(number_data)
    plt.show()
    return mfcc_matrix

frame_length, mel_filters, predictor_order, cepstral_coefficient, GMM_components, covariance_type, permutations = setup()
Fs, data = import_files('train')
compute_mfcc(Fs, data, frame_length)