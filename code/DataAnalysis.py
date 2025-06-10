import os, math, random, time, utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


### compare the probability density using a bar chart
def compare_density(data_name, data1, data2, num_sample, font_size=32):
    plt.figure(figsize=(16, 9))

    # divide the entire area into 100 equal parts
    intervals = np.linspace(0, 1, 101)
    bin_centers = [(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)]

    plt.bar(bin_centers, data1, width=0.009, color="blue", label="$P_m$", edgecolor="azure")
    plt.bar(bin_centers, data2, width=0.009, color="red", label="$Q_m$", edgecolor="azure")

    # set chart properties
    plt.title('{} samples'.format(num_sample), fontsize=font_size)
    plt.xlabel(r"$\epsilon$", fontsize=font_size)
    plt.ylabel('Probability', fontsize=font_size)
    plt.legend(loc='upper right', fontsize=font_size)
    plt.ylim(0, 1.01)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # scale font
    plt.tick_params(axis='both', which='major', labelsize=font_size)

    # save picture
    plt.savefig('../prediction_result/{}Sample{}.png'.format(data_name, num_sample), dpi=100)
    plt.close()


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


### compare mean/std of distribution using a line chart
def compare_distribution(data_name, data1, data2, name, span=10, y_scale=[-0.01, 1.01], font_size=32):
    plt.figure(figsize=(16, 9))

    x_list = range(1, len(data1) + 1)
    plt.plot(x_list, data1, color='blue', linestyle='-', label="$P_m$", linewidth=3)
    plt.plot(x_list, data2, color='red', linestyle='-', label="$Q_m$", linewidth=3)

    plt.legend()
    plt.xticks(list(range(0, len(data1) + 1, span)), fontsize=font_size)
    plt.xlabel("Iterations", fontsize=font_size)
    if name == 'mean':
        plt.ylabel('Mean', fontsize=font_size)
    else:
        plt.ylabel('Standard deviation', fontsize=font_size)
    plt.ylim(y_scale[0], y_scale[1])
    plt.yticks(fontsize=font_size)
    plt.legend(loc='upper right', fontsize=font_size)

    # save picture
    plt.savefig('../prediction_result/{}_compare_{}.png'.format(data_name, name), dpi=100)
    plt.close()


### compare Wasserstein distance using a line chart
def compare_wasserstein(data_name, data, span=10, font_size=32):
    plt.figure(figsize=(16, 9))

    x_list = range(1, len(data) + 1)
    y = data
    plt.plot(x_list, y, color='orange', linestyle='-', label='$W(P_m , Q_m)$', linewidth=3)
    # plt.scatter(x_list, y, color='purple', marker='o', s=20, alpha=0.8)

    plt.legend()
    plt.xticks(list(range(0, len(data) + 1, span)), fontsize=font_size)
    plt.xlabel("Iterations", fontsize=font_size)
    plt.ylabel("Wasserstein distance", fontsize=font_size)
    plt.ylim(-0.01, 1.01)
    plt.yticks(fontsize=font_size)
    plt.legend(loc='upper right', fontsize=font_size)

    # save picture
    plt.savefig('../prediction_result/{}_WD.png'.format(data_name), dpi=100)
    plt.close()


class Dataset:
    def __init__(self, data_name, num_mini_sample, num_round=50, num_repeat=100, num_span=100):
        self.data_name = data_name
        self.num_mini_sample = num_mini_sample
        self.num_round = num_round
        self.num_repeat = num_repeat
        self.num_span = num_span
        self.path = '../user_data/tmp_data/'

        # find the corresponding distribution
        self.sub_name = '({},{},{},{})'.format(self.num_repeat, self.num_span, self.num_mini_sample, self.num_round)
        self.P = self.search_file('{}_P{}'.format(self.data_name, self.sub_name))
        self.Q = self.search_file('{}_Q{}'.format(self.data_name, self.sub_name))

    def search_file(self, name):
        files = utils.find_files(self.path, name)
        if len(files) > 0:
            result = utils.load_file(self.path + files[0])
        else:
            print("Warning: There is no file to be matched.")
            result = []

        return result

    ### analyze the P_m and Q_m distributions
    def analyze_dis(self, list=[1]):
        for i in list:
            data1 = self.P[i - 1]
            data2 = self.Q[i - 1]
            compare_density(self.data_name, data1, data2, i * self.num_mini_sample)

    ### analyze mean/standard deviation of the distributions
    def analyze_statistic(self):
        P_mean, P_std = calculate_mean_std(self.P, self.num_span)
        Q_mean, Q_std = calculate_mean_std(self.Q, self.num_span)
        compare_distribution(self.data_name, P_mean, Q_mean, 'mean', span=10, y_scale=[-0.01, 1.01])
        compare_distribution(self.data_name, P_std, Q_std, 'std', span=10, y_scale=[-0.001, 0.055])

    ### analyze the wasserstein distance
    def analyze_wasserstein(self):
        dists = [i for i in np.linspace(0, 1, 100)]
        WD = []
        for i in range(len(self.P)):
            wd = scipy.stats.wasserstein_distance(dists, dists, np.array(self.P[i]), np.array(self.Q[i]))
            WD.append(wd)

        compare_wasserstein(self.data_name, WD, span=10)


if __name__ == '__main__':
    print('start')

    CBL = Dataset('CBL', num_mini_sample=25, num_round=50, num_repeat=1000, num_span=100)
    CBL.analyze_dis([1, 2, 5, 10, 20, 50])
    CBL.analyze_statistic()
    CBL.analyze_wasserstein()

    TH = Dataset('TH', num_mini_sample=200, num_round=50, num_repeat=1000, num_span=100)
    TH.analyze_dis([1, 2, 5, 10, 25, 50])
    TH.analyze_statistic()
    TH.analyze_wasserstein()

    Iris = Dataset('Iris', num_mini_sample=200, num_round=50, num_repeat=100, num_span=100)
    Iris.analyze_dis([1, 2, 5, 10, 25, 50])
    Iris.analyze_statistic()
    Iris.analyze_wasserstein()

    print('end')
