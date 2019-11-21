#from eval import evaluate
import os
from scipy.io.wavfile import read
import numpy as np
import python_speech_features as sp
import matplotlib.pyplot as plt
import sklearn.model_selection


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


def compute_mfcc(folder, frame_length):
    data = {}
    mfcc_dict = {} #default dict, sprawdzić, iteracja po kluczach, collections
    # Creating a vector of arrays with waves for every number
    for file in os.listdir(folder):
        if file[6] == '0':
            mfcc_dict[file[0:5]] = {}
        Fs, wavefile = read(folder + '/' + file)
        data[file] = {'data': wavefile, "Fs": Fs}
        mfcc_data = sp.base.mfcc(data[file]["data"], samplerate=data[file]["Fs"], winlen=frame_length, lowfreq=50,
                            highfreq=8000, winstep=0.01, appendEnergy=True, nfft=882)
        mfcc_dict[file[0:5]][file[6]] = mfcc_data
    print(mfcc_dict)
    return mfcc_dict

def split_data(mfcc_matrix, number_splits):
    mfcc_matrix_train = []  # mfcc_matrix_train [digit][number_of_sets][number_of_mfcc][vector_of_mcc][element]
    mfcc_matrix_test = []
    folds = sklearn.model_selection.KFold(n_splits=number_splits)
    folds.get_n_splits(mfcc_matrix)
    data_matrix_train = []
    data_matrix_test = []
    for train_index, test_index in folds.split(mfcc_matrix):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = mfcc_matrix[train_index], mfcc_matrix[test_index]




def concatenate_data(mfcc_matrix_set):
    mfcc_concatenated_matrix = []
    for i in range(10):
        mfcc_concatenate = 0
        for j in range(len(mfcc_matrix_set[i])):  # iterating through number of sets joining mfcc vectors
            print(mfcc_matrix_set[i][j][:][0])
            mfcc_concatenate = np.concatenate(mfcc_concatenate, mfcc_matrix_set[i][j][:][0])
        print(mfcc_concatenate)
        mfcc_concatenated_matrix.append(mfcc_concatenate)
    return mfcc_concatenated_matrix  # [digit][number of set][c column][element]


def create_train_GMMmodels(mfcc_matrix_train_concatenated, mfcc_matrix_test_concatenated, n_components):
    gmm_train_vector = []
    gmm_test_vector = []
    for i in range(10):
        gmm_training_results = []
        gmm_testing_results = []
        for j in range(len(mfcc_matrix_train_concatenated[i])):  # iteration through number of sets
            gmm_temporary = sklearn.mixture.gaussian_mixture(n_components=n_components, max_iter=100,
                                                             random_state=4)  # creating temporary gmm
            gmm_training_results.append(gmm_temporary.sklearn.mixture.gaussian_mixture.fit(
                mfcc_matrix_train_concatenated[i][j]))  # training temporary gmm
            gmm_testing_results.append(gmm_temporary.sklearn.mixture.gaussian_mixture.score(
                mfcc_matrix_test_concatenated[i][j]))  # testing temporary gmm
        gmm_train_vector.append(gmm_training_results)  # writing training results
        gmm_test_vector.append(gmm_testing_results)  # writing testing results
    return gmm_train_vector, gmm_test_vector  # [digit][number of sets][results]


frame_length, mel_filters, predictor_order, cepstral_coefficient, GMM_components, covariance_type, permutations = setup()
mfcc_data = compute_mfcc('train', frame_length)
mfcc_matrix_train, mfcc_matrix_test = split_data(mfcc_data, 5)
mfcc_matrix_train_concatenated = concatenate_data(mfcc_matrix_train)
mfcc_matrix_test_concatenated = concatenate_data(mfcc_matrix_test)
gmm_train_vector, gmm_test_vector = create_train_GMMmodels(mfcc_matrix_train_concatenated,
                                                           mfcc_matrix_test_concatenated, 100)
