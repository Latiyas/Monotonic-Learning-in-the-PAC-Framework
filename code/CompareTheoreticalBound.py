import os, math, random, time, utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


### calculate the theoretical probability density function - Q (realizability assumption)
def GetTPDF_R(num_sample, num_span, num_byte):
    q = [0.0] * num_span
    H = 3 ** num_byte

    ### realizability assumption
    th = math.log(H) / num_sample
    # print(th)
    pro_sum = 0.0
    for i in range(num_span - 1):
        x = (i + 1.0) / num_span
        if x >= th:
            tmp = 1 - H * (math.exp(-1 * num_sample * x))
            q[i] = tmp - pro_sum
            pro_sum = tmp
    q[num_span - 1] = 1 - pro_sum

    return q


### calculate the theoretical probability density function - Q (agnostic case)
def GetTPDF_A(num_sample, num_span, num_byte):
    q = [0.0] * num_span
    H = 3 ** num_byte

    ### agnostic case
    th = math.sqrt(2 * math.log(2 * H) / num_sample)
    # print(th)
    pro_sum = 0.0
    for i in range(num_span - 1):
        x = (i + 1.0) / num_span
        if x >= th:
            tmp = 1 - 2 * H * (math.exp(-1 * num_sample * (x ** 2) / 2))
            q[i] = tmp - pro_sum
            pro_sum = tmp
    q[num_span - 1] = 1 - pro_sum

    return q


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
    plt.plot(x_list, data1, color='green', linestyle='-', label="$Q_m^{Realizability}$", linewidth=3)
    plt.plot(x_list, data2, color='purple', linestyle='-', label="$Q_m^{Agnostic}$", linewidth=3)

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
    plt.savefig('../prediction_result/{}_compare_{}_CTB.png'.format(data_name, name), dpi=100)
    plt.close()


### compare Wasserstein distance using a line chart
def compare_wasserstein(data_name, data, span=10, font_size=32):
    plt.figure(figsize=(16, 9))

    x_list = range(1, len(data) + 1)
    y = data
    plt.plot(x_list, y, color='orange', linestyle='-', label='$W(Q_m^{Realizability}, Q_m^{Agnostic})$', linewidth=3)
    # plt.scatter(x_list, y, color='purple', marker='o', s=20, alpha=0.8)

    plt.legend()
    plt.xticks(list(range(0, len(data) + 1, span)), fontsize=font_size)
    plt.xlabel("Iterations", fontsize=font_size)
    plt.ylabel("Wasserstein distance", fontsize=font_size)
    plt.ylim(-0.01, 1.01)
    plt.yticks(fontsize=font_size)
    plt.legend(loc='upper right', fontsize=font_size)

    # save picture
    plt.savefig('../prediction_result/{}_WD_CTB.png'.format(data_name), dpi=100)
    plt.close()


### analyze mean/standard deviation of the distributions
def analyze_statistic(Q_R, Q_A, num_span):
    Q_R_mean, Q_R_std = calculate_mean_std(Q_R, num_span)
    Q_A_mean, Q_A_std = calculate_mean_std(Q_A, num_span)
    compare_distribution('CBL', Q_R_mean, Q_A_mean, 'mean', span=10, y_scale=[-0.01, 1.01])
    compare_distribution('CBL', Q_R_std, Q_A_std, 'std', span=10, y_scale=[-0.001, 0.055])

### analyze the wasserstein distance
def analyze_wasserstein(Q_R, Q_A):
    dists = [i for i in np.linspace(0, 1, 100)]
    WD = []
    for i in range(len(Q_R)):
        wd = scipy.stats.wasserstein_distance(dists, dists, np.array(Q_R[i]), np.array(Q_A[i]))
        WD.append(wd)

    compare_wasserstein('CBL', WD, span=10)


if __name__ == '__main__':
    print('start')

    # number of Booleans
    num_byte = 10
    # number of intervals
    num_span = 100
    # minimal sample size
    num_mini_sample = 25
    # number of iterations
    num_round = 50

    Q_R_list = []
    Q_A_list = []
    for i in range(num_round):
        num_sample = num_mini_sample * (i + 1)
        Q_R = GetTPDF_R(num_sample, num_span, num_byte)
        Q_A = GetTPDF_A(num_sample, num_span, num_byte)
        Q_R_list.append(Q_R)
        Q_A_list.append(Q_A)

    analyze_statistic(Q_R_list, Q_A_list, num_span)
    analyze_wasserstein(Q_R_list, Q_A_list)

    print('end')
