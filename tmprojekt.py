import os
from scipy.io.wavfile import read
import numpy as np
import python_speech_features as sp
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.mixture as mix
import math
import csv
import eval
import json


def setup():
    with open('setup.json') as f:
        data = json.load(f)
    frame_length = data['Frame length']
    mel_filters = data['Mel filters amount']
    predictor_order = data['Predictor order']
    cepstral_coefficient = data['Cepstral coefficients amount']
    GMM_components = data['GMM components amount']
    covariance_type = data['Covariance matrix type']
    return frame_length, mel_filters, predictor_order, cepstral_coefficient, GMM_components, covariance_type

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
            if file[5:8] == '_{}_'.format(i):
                rate, data = read(folder + '/' + file)
                Fs[i].append(rate)
                mfcc_data = sp.base.mfcc(data, samplerate=rate, winlen=frame_length, lowfreq=50, nfilt=mel_filters, ceplifter = 0, highfreq=8000, winstep=0.01, appendEnergy=True, nfft=1024)
                #calculating MFCC of audio file
                mfcc[i].append(np.transpose(mfcc_data)) #transposing mfcc data
                filename[i].append(file)
    return Fs, mfcc, filename
    #creating a vector of vectors of MFCC arrays
    #filename [digit][speaker_name]

def split_data(mfcc_matrix, number_splits): #mfcc_matrix [digit][speaker_number][c column][element]
    mfcc_matrix_train = []  # mfcc_matrix_train [digit][number_of_sets][number_of_mfcc][vector_of_mcc][element]
    mfcc_matrix_test = []
    folds = sklearn.model_selection.KFold(n_splits=number_splits)
    range_of_iteration = 0
    vector_of_training_sets = []
    vector_of_testing_sets = []
    for train_index, test_index in folds.split(mfcc_matrix[0]):
        vector_of_training_sets.append(list(train_index))
        vector_of_testing_sets.append(list(test_index))
        range_of_iteration += 1


    for i in range(10):
        data_matrix_train = []
        data_matrix_test = []
        for k in range(range_of_iteration):
            data_vector_train = []
            data_vector_test = []
            for j in vector_of_training_sets[k]:  # creating vector of mfcc arrays
                data_vector_train.append(mfcc_matrix[i][j])
            for j in vector_of_testing_sets[k]:  # creating vector of mfcc arrays
                data_vector_test.append(mfcc_matrix[i][j])
            data_matrix_train.append(data_vector_train)
            data_matrix_test.append(data_vector_test)
        mfcc_matrix_train.append(data_matrix_train)  # creating train array for each numer
        mfcc_matrix_test.append(data_matrix_test)
    return mfcc_matrix_train, mfcc_matrix_test  # returning vector of train, and test data[digit][number of set][number of speaker][vector_of_mfcc][element]


def concatenate_data(mfcc_matrix_set):
    mfcc_concatenated_matrix = []
    for i in range(10):
        mfcc_concatenated_sets = []
        for j in range(len(mfcc_matrix_set[i])): #iterating through number of sets joining mfcc vectors
            #mfcc_concatenate = []
            for k in range(len(mfcc_matrix_set[i][j])):#iterationg through number of mfcc

                if k == 0:
                    mfcc_concatenate = np.asarray(mfcc_matrix_set[i][j][k])
                else:
                    mfcc_concatenate = np.concatenate((mfcc_concatenate, np.asarray(mfcc_matrix_set[i][j][k])),1) #sklejenie tylko jaki axis ?
            mfcc_concatenated_sets.append(mfcc_concatenate)
        mfcc_concatenated_matrix.append(mfcc_concatenated_sets)
    return mfcc_concatenated_matrix #[digit][number of set][c column][element]


def create_train_GMMmodels(mfcc_matrix_train_concatenated, mfcc_matrix_test, n_components, covariance_type):
    GMM_models = []  # [digit][number of set (model)]
    for i in range(10):  # creating a vector of trained gmm models
        gmm_models = []
        for j in range(len(mfcc_matrix_train_concatenated[i])):  # iteration through number of sets
            gmm_temporary = mix.GaussianMixture(n_components=n_components, max_iter=100, random_state=2,
                                                covariance_type=covariance_type)
            gmm_models.append(gmm_temporary.fit(np.transpose(mfcc_matrix_train_concatenated[i][j])))
        GMM_models.append(gmm_models)
    general_statistics_final = []

    for i in range(10):
        number_statistics = []
        likelihood_iter = 0
        for j in range(len(mfcc_matrix_train_concatenated[i])):  # iteration through number of sets
            likelihood_result = 0  # temporary array of proper match 0 and 1

            for k in range(len(mfcc_matrix_test[i][j])):  # iteration through number of speaker
                likelihood = []  # temporary array of likelihood for every model
                for l in range(len(GMM_models)):  # iteration through number of gmm
                    likelihood.append(GMM_models[l][k].score(np.transpose(mfcc_matrix_test[i][j][k])))
                if likelihood.index(max(likelihood)) == i:
                    likelihood_result += 1
                    likelihood_iter += 1
                else:
                    likelihood_iter += 1
            number_statistics.append(likelihood_result)

        general_statistics_final.append(sum(number_statistics) / likelihood_iter)
    print(general_statistics_final)

def concatenate_original_data(mfcc_matrix):
    mfcc_concatenated_matrix = []
    for i in range(10):
        mfcc_concatenated = []
        for j in range(len(mfcc_matrix[i])):  # iterating through number of sets joining mfcc vectors
            if j == 0:
                mfcc_concatenated = mfcc_matrix[i][j]
            else:
                mfcc_concatenated = np.concatenate((mfcc_concatenated, mfcc_matrix[i][j]), axis=1)  # sklejenie tylko jaki axis ?
        mfcc_concatenated_matrix.append(mfcc_concatenated)
    return mfcc_concatenated_matrix  # [digit][c column][element]

def create_model(mfcc_matrix_concatenated, n_components, covariance_type):
    gmm_numbers = []
    for i in range(10):
        gmm = mix.GaussianMixture(n_components = n_components, max_iter=100, random_state=2, covariance_type=covariance_type)
        gmm.fit(np.transpose(mfcc_matrix_concatenated[i]))
        gmm_numbers.append(gmm)
    return gmm_numbers

def evaluation(folder, gmm):
    with open('evaluation.txt', 'w', newline='') as csv_file:
        csv_object = csv.writer(csv_file)
        for file in os.listdir(folder):
            loglikehood = []
            rate, data = read(folder + '/' + file)
            mfcc_data = sp.base.mfcc(data, samplerate=rate, winlen=frame_length, lowfreq=50, nfilt=mel_filters,
                                     ceplifter=0, highfreq=8000, winstep=0.01, appendEnergy=True, nfft=1024)
            for i in range(10):
                loglikehood.append(gmm[i].score(mfcc_data))
            max = np.amax(loglikehood)
            argmax = np.argmax(loglikehood)
            csv_object.writerow([file, argmax, round(max, 2)])
    csv_file.close()

frame_length, mel_filters, predictor_order, cepstral_coefficient, GMM_components, covariance_type = setup()
Fs, mfcc_matrix, filename = compute_mfcc('train', mel_filters)
mfcc_matrix_train, mfcc_matrix_test = split_data(mfcc_matrix, 5)
mfcc_matrix_train_concatenated = concatenate_data(mfcc_matrix_train)
create_train_GMMmodels(mfcc_matrix_train_concatenated, mfcc_matrix_test, GMM_components, covariance_type)
mfcc_matrix_concatenated = concatenate_original_data(mfcc_matrix)
gmm_numbers = create_model(mfcc_matrix_concatenated, GMM_components, covariance_type)
evaluation('eval', gmm_numbers)
eval.evaluate('evaluation.txt')
