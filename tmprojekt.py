import os
from scipy.io.wavfile import read
import numpy as np
import python_speech_features as sp
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.mixture as mix
import math

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

def compute_mfcc(folder, mel_filters):
    Fs = []
    mfcc = []
    filename = []
    #creating a vector of vectors of MFCC arrays [digit][speaker_number][c column][element]
    for i in range(10):
        Fs.append([])
        mfcc.append([])
        filename.append([])
        for file in os.listdir(folder):
            if file[5:8] == '_%d_' % i:
                rate, data = read(folder + '/' + file)
                Fs[i].append(rate)
                mfcc_data = sp.base.mfcc(data, samplerate=rate, winlen=frame_length, lowfreq=50, nfilt=mel_filters,
                                         ceplifter = 0, highfreq=8000, winstep=0.01, appendEnergy=True, nfft=1024)
                mfcc[i].append(mfcc_data)
                filename[i].append(file)
    return Fs, mfcc, filename
    #creating a vector of vectors of MFCC arrays [digit][speaker_number][c column][element]
    #filename [digit][speaker_name]

def split_data(mfcc_matrix, number_splits):
    mfcc_matrix_train = [] # mfcc_matrix_train [digit][number_of_sets][number_of_mfcc][vector_of_mcc][element]
    mfcc_matrix_test = []
    for i in range(10):
        folds = sklearn.model_selection.KFold(n_splits=number_splits)
        folds.get_n_splits(mfcc_matrix[i])
        data_matrix_train = []
        data_matrix_test = []
        for train_index, test_index in folds.split(mfcc_matrix[i]):
            data_vector_train = []
            data_vector_test = []
            for j in train_index: #creating vector of mfcc arrays
                data_vector_train.append(mfcc_matrix[i][j])
            for j in test_index: #creating vector of mfcc arrays
                data_vector_test.append(mfcc_matrix[i][j])
            data_matrix_train.append(data_vector_train) #adding vector of mfcc to training matrix
            data_matrix_test.append(data_vector_test) #adding vector of mfcc to testing matrix
        mfcc_matrix_train.append(data_matrix_train) #creating train array for each numer
        mfcc_matrix_test.append(data_matrix_test)
    return mfcc_matrix_train, mfcc_matrix_test #returning vector of train, and test data[digit][number of set][number of mfcc][vector_of_mfcc][element]

def concatenate_data(mfcc_matrix_set):
    mfcc_concatenated_matrix = []
    for i in range(10):
        mfcc_concatenated_sets = []
        for j in range(len(mfcc_matrix_set[i])): #iterating through number of sets joining mfcc vectors
            #mfcc_concatenate = []
            for k in range(len(mfcc_matrix_set[i][j])):#iterationg through number of mfcc
                #np.array(mfcc_matrix_set[i][j])
                #a = np.array(a)
                if k == 0:
                    mfcc_concatenate = mfcc_matrix_set[i][j][k]
                else:
                    mfcc_concatenate = np.concatenate((mfcc_concatenate, mfcc_matrix_set[i][j][k]), axis = 0)#sklejenie tylko jaki axis ?
            mfcc_concatenated_sets.append(mfcc_concatenate)
        mfcc_concatenated_matrix.append(mfcc_concatenated_sets)
    return mfcc_concatenated_matrix #[digit][number of set][c column][element]

def create_train_GMMmodels(mfcc_matrix_train_concatenated, mfcc_matrix_test, n_components, covariance_type):
    for i in range(10):
        for j in range(len(mfcc_matrix_train_concatenated[i])):#iteration through number of sets
            gmm_temporary = mix.GaussianMixture(n_components = n_components, max_iter=100, random_state=4, covariance_type=covariance_type)
            gmm_temporary.fit(mfcc_matrix_train_concatenated[i][j])
            likelihood_array = []
            for k in range(len(mfcc_matrix_test[i][j])):
                likelihood = math.exp((gmm_temporary.score(mfcc_matrix_test[i][j][k])/len(mfcc_matrix_test[i][j][k])))
                likelihood_array.append(likelihood)
            average_likelihood = np.mean(likelihood_array)*100
            stdev_likelihood = np.std(likelihood_array)*100
            print("Dla liczby %d, próby %d podobieństwo wynosi: %f +/- %f" % (i, j, average_likelihood, stdev_likelihood))

def concatenate_original_data(mfcc_matrix):
    mfcc_concatenated_matrix = []
    for i in range(10):
        mfcc_concatenated = []
        for j in range(len(mfcc_matrix[i])):  # iterating through number of sets joining mfcc vectors
            if j == 0:
                mfcc_concatenated = mfcc_matrix[i][j]
            else:
                mfcc_concatenated = np.concatenate((mfcc_concatenated, mfcc_matrix[i][j]), axis=0)  # sklejenie tylko jaki axis ?
        mfcc_concatenated_matrix.append(mfcc_concatenated)
    return mfcc_concatenated_matrix  # [digit][c column][element]

def create_model(mfcc_matrix_concatenated, mfcc_matrix_test, n_components, covariance_type):
    gmm_numbers = []
    for i in range(10):
        gmm = mix.GaussianMixture(n_components = n_components, max_iter=100, random_state=4, covariance_type=covariance_type)
        gmm.fit(mfcc_matrix_concatenated[i])
        gmm_numbers.append(gmm)
    return gmm_numbers

def evaluation(folder, gmm):
    for file in os.listdir(folder):
        rate, data = read(folder + '/' + file)
        mfcc_data = sp.base.mfcc(data, samplerate=rate, winlen=frame_length, lowfreq=50, nfilt=mel_filters,
                                 ceplifter=0, highfreq=8000, winstep=0.01, appendEnergy=True, nfft=1024)
        for i in range(10):
            print(gmm[i].score(mfcc_data))

frame_length, mel_filters, predictor_order, cepstral_coefficient, GMM_components, covariance_type, permutations = setup()
Fs, mfcc_matrix, filename = compute_mfcc('train', mel_filters)
mfcc_matrix_train, mfcc_matrix_test = split_data(mfcc_matrix, 5)
mfcc_matrix_train_concatenated = concatenate_data(mfcc_matrix_train)
#create_train_GMMmodels(mfcc_matrix_train_concatenated, mfcc_matrix_test, GMM_components, covariance_type)
mfcc_matrix_concatenated = concatenate_original_data(mfcc_matrix)
gmm_numbers = create_model(mfcc_matrix_concatenated, mfcc_matrix_test, GMM_components, covariance_type)
evaluation('test', gmm_numbers)
