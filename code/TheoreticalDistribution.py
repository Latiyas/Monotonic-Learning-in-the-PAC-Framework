import os, math, random, time, utils
import numpy as np
import matplotlib.pyplot as plt


### obtain the cumulative distribution function
def CDF(H, m):
    e = math.exp(1)
    # finite hypothesis space, consistent case
    X = np.linspace(0.0, 1.025, 10000, endpoint=True)
    Y = 1 - 2 * H * (e ** (-1 * m * (X ** 2) / 2))

    Z = []
    for i in range(len(Y)):
        if X[i] >= 1:
            Z.append(1)
        elif Y[i] < 0:
            # print(X[i])
            Z.append(0)
        else:
            Z.append(Y[i])

    Z = np.array(Z)

    # create a new graphics window
    plt.figure(figsize=(8, 6))
    # main curve
    plt.plot(X, Z, color='red', linewidth=2.5, label="m={}".format(m))
    # legend position and font size
    plt.legend(loc='upper left', fontsize=24)
    # filler
    plt.fill_between(X, 0, Z, color='pink', alpha=0.4)
    # scale font
    plt.tick_params(axis='both', which='major', labelsize=28)

    plt.xlabel(r"$\epsilon$", fontsize=32)
    plt.ylabel(r"$F_m(\epsilon)$", fontsize=32)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # high-resolution
    plt.savefig('../prediction_result/distribution_H09_m{}.png'.format(m), dpi=100)
    # plt.show()


### calculate the theoretical probability density function - Q (finite hypothesis space)
def GetTPDF(H, m, num_span):
    # ### realizability assumption
    # q = [0.0] * num_span
    # th = math.log(H) / m
    # # print(th)
    # pro_sum = 0.0
    # for i in range(num_span - 1):
    #     x = (i + 1.0) / num_span
    #     if x >= th:
    #         tmp = 1 - H * (math.exp(-1 * m * x))
    #         q[i] = tmp - pro_sum
    #         pro_sum = tmp
    # q[num_span - 1] = 1 - pro_sum

    ### agnostic case
    q = [0.0] * num_span
    th = math.sqrt(2 *  math.log(2 * H) / m)
    # print(th)
    pro_sum = 0.0
    for i in range(num_span - 1):
        x = (i + 1.0) / num_span
        if x >= th:
            tmp = 1 - 2 * H * (math.exp(-1 * m * (x ** 2) / 2))
            q[i] = tmp - pro_sum
            pro_sum = tmp
    q[num_span - 1] = 1 - pro_sum

    return q


### obtain the probability density function
def PDF(data1, data2, data3):
    # divide the entire area into 100 equal parts
    intervals = np.linspace(0, 1, 101)
    bin_centers = [(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)]

    # statistics for the three data sets
    counts1 = np.array(data1)
    counts2 = np.array(data2)
    counts3 = np.array(data3)

    plt.figure(figsize=(16, 8))

    # plot bar charts (overlapping plots)
    plt.bar(bin_centers, counts1, width=0.009, color="orange", label="m=35", alpha=0.6, edgecolor="black")
    plt.bar(bin_centers, counts2, width=0.009, color="green", label="m=70", alpha=0.6, edgecolor="black")
    plt.bar(bin_centers, counts3, width=0.009, color="purple", label="m=150", alpha=0.6, edgecolor="black")

    plt.xlabel(r"$\epsilon$", fontsize=24)
    plt.ylabel("Probability", fontsize=24)

    # adjust scale
    plt.xticks(np.linspace(0, 1, 11), fontsize=24)
    plt.yticks(fontsize=24)
    # scale font
    plt.tick_params(axis='both', which='major', labelsize=24)

    plt.legend(fontsize=24, loc="upper right")

    # add a grid
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # layout adjustment
    plt.tight_layout()

    plt.savefig('../prediction_result/density_H09.png', dpi=300)
    # plt.show()


if __name__ == '__main__':
    print('start')

    # number of hypotheses
    H = 1000000
    # number of samples
    m_list = [35, 70, 150]

    for m in m_list:
        # obtain a cumulative distribution function
        CDF(H, m)

    # number of intervals
    num_span = 100

    Q_den = []
    for m in m_list:
        Q_den.append(GetTPDF(H, m, num_span))

    # obtain a probability density function
    PDF(Q_den[0], Q_den[1], Q_den[2])

    print('end')
