import os, math, random, time, utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


### calculate the frequency of monotonic updatas
def cal_monotone(data):
    n_non_mono = 0
    for i in range(1, len(data)):
        if data[i - 1] < data[i]:
            n_non_mono += 1
    fre_mono = (len(data) - n_non_mono) / len(data)
    return fre_mono


### calculate mean and standard deviation
def calculate_mean_std(data, num_span):
    loss_mean = []
    loss_std = []
    for i in range(len(data)):
        loss_list = data[i]
        tmp_mean = 0
        tmp_mean_square = 0
        for j in range(len(loss_list)):
            x = (j + j + 1) / 2 / num_span
            tmp_mean += loss_list[j] * x
            tmp_mean_square += loss_list[j] * x * x
        loss_mean.append(tmp_mean)
        loss_std.append(pow(tmp_mean_square - tmp_mean * tmp_mean, 0.5))
    return loss_mean, loss_std


class Dataset:
    def __init__(self, data_name, num_mini_sample, num_round=50, num_repeat=100, num_span=100):
        self.data_name = data_name
        self.num_mini_sample = num_mini_sample
        self.num_round = num_round
        self.num_repeat = num_repeat
        self.num_span = num_span
        self.path = '../user_data/tmp_data/'

    def search_file(self, name):
        files = utils.find_files(self.path, name)
        if len(files) > 0:
            result = utils.load_file(self.path + files[0])
        else:
            print("Warning: There is no file to be matched.")
            result = []

        return result

    ### analyze the monotone of the distributions
    def analyze_monotone(self, name):
        P = utils.load_file(self.path + '{}'.format(name))
        P_mean, P_std = calculate_mean_std(P, self.num_span)

        fre_mono = cal_monotone(P_mean)
        print('monotone: {}'.format(fre_mono))


if __name__ == '__main__':
    print('start')

    CBL = Dataset('CBL', num_mini_sample=5, num_round=50, num_repeat=100, num_span=100)

    files = utils.find_files(CBL.path, '({},{},{},{})'.format(CBL.num_repeat, CBL.num_span, CBL.num_mini_sample, CBL.num_round))
    if len(files) > 0:
        CBL_name = files[0]
    else:
        print("Warning: There is no file to be matched.")
        exit()

    CBL.analyze_monotone(CBL_name)

    print('end')
