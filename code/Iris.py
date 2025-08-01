import os, math, random, time, utils
import numpy as np
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse


### parsing and configuration
def parse_args():
    desc = "Monotonicity analysis on the Iris classification problem"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--num_repeat', type=int, default=1000, help='The number of repeated sampling')
    parser.add_argument('--num_span', type=int, default=100, help='The number of intervals')
    parser.add_argument('--num_mini_sample', type=int, default=200, help='The minimum sample size')
    parser.add_argument('--num_iter', type=int, default=50, help='The number of iterations')
    parser.add_argument('--tmp_dir', type=str, default='../user_data/tmp_data/',
                        help='Directory name to save temp file')

    return check_args(parser.parse_args())


### checking arguments
def check_args(args):
    # num_repeat
    try:
        assert args.num_repeat >= 1
    except:
        print('The number of repeated samplings should be no less than 1')

    # num_span
    try:
        assert args.num_span >= 1
    except:
        print('The number of intervals should be no less than 1')

    # num_mini_sample
    try:
        assert args.num_mini_sample >= 1
    except:
        print('The minimum sample size should be no less than 1')

    # num_iter
    try:
        assert args.num_iter >= 1
    except:
        print('The number of iterations should be no less than 1')

    # tmp_dir
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    return args


### outputting config
def output_config(args):
    print('The number of repeated samplings:{}'.format(args.num_repeat))
    print('The number of intervals:{}'.format(args.num_span))
    print('The minimum sample size:{}'.format(args.num_mini_sample))
    print('The number of iterations:{}'.format(args.num_iter))


### load iris data set
def load_iris_train_test(ratio=0.8, types=[1, 2]):
    # iris data set
    iris = datasets.load_iris()

    instances = []
    labels = []
    for x, y in zip(iris.data, iris.target):
        if y in types:
            instances.append(x)
            labels.append(y)

    test_num = int(len(instances) - len(instances) * ratio)

    X_train, X_test, y_train, y_test = train_test_split(instances, labels, test_size=test_num, random_state=53,
                                                        stratify=labels)

    return X_train, y_train, X_test, y_test


# generate sample index
def generate_sample_index(num_train, num_sample):
    # the sample index of training set
    # random sampling (put back)
    train_slice = list(np.random.randint(0, num_train, num_sample))

    return train_slice


### generate the samples
def generate_samples(data_x, data_y, slice):
    instances = np.array([data_x[i] for i in slice])
    labels = np.array([data_y[i] for i in slice])

    return instances, labels


### get the hypothesis
def get_hypothesis(instances, labels):
    # SVM regularization parameter
    C = 10
    model = svm.SVC(kernel="linear", C=C)

    hyp = model.fit(instances, labels)

    return hyp


### loss function
def loss_function(hyp, data_x, data_y):
    y_pred = hyp.predict(data_x)
    result = 1 - accuracy_score(data_y, y_pred)

    return result


### Calculate the minimum generalization loss
def calculate_min_loss(data_x, data_y, types=[1, 2]):
    # import some data to play with
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    instances = []
    labels = []
    for x, y in zip(iris.data, iris.target):
        if y in types:
            instances.append(x)
            labels.append(y)

    # X = iris.data[:, :4]
    # Y = iris.target

    # SVM regularization parameter
    C = 2
    model = svm.SVC(kernel="linear", C=C)

    hyp = model.fit(instances, labels)

    min_loss = loss_function(hyp, data_x, data_y)
    # print(min_loss)

    return min_loss


### training the model
def Train(trainsets_x, trainsets_y, testsets_x, testsets_y, num_sample, num_repeat=1):
    loss_list = []
    for i in range(num_repeat):
        train_slice = generate_sample_index(len(trainsets_y), num_sample)
        instances, labels = generate_samples(trainsets_x, trainsets_y, train_slice)
        hyp = get_hypothesis(instances, labels)
        test_loss = loss_function(hyp, testsets_x, testsets_y)
        # print("epoch {}: {}".format(i, test_loss))
        loss_list.append(test_loss)

    return loss_list


### calculate the empirical probability density function - P
def GetEPDF(loss_list, num_span):
    # use np.histogram to get the distribution of frequencies
    p_num, _ = np.histogram(loss_list, bins=num_span, range=(0, 1))
    num_repeat = len(loss_list)
    p = [x / num_repeat for x in p_num]

    return p


### recalibrate the distribution
def Calibration_Distribution(ori_dis, L_min, num_span):
    # ori_dis = list(range(100))
    # num_empty = 5
    # num_empty = int(L_min * num_span)
    num_empty = math.ceil(int(L_min * num_span * 1000) / 1000)
    cal_dis = [0.0] * num_empty + ori_dis[:num_span - num_empty]
    cal_dis[-1] = cal_dis[-1] + sum(ori_dis[num_span - num_empty:])

    return cal_dis


### calculate the theoretical probability density function - Q (finite VC dimension)
def GetTPDF(L_min, num_sample, num_span, num_VC):
    q = [0.0] * num_span
    H = (2 * math.e * num_sample / num_VC) ** num_VC
    th = math.sqrt(32 * (num_VC * math.log(2 * math.e * num_sample / num_VC) + math.log(4)) / num_sample)
    # print(th)
    pro_sum = 0.0
    for i in range(num_span - 1):
        x = (i + 1.0) / num_span
        if x >= th:
            tmp = 1 - 4 * H * math.exp(-1 * num_sample * (x ** 2) / 32)
            q[i] = tmp - pro_sum
            pro_sum = tmp
    q[num_span - 1] = 1 - pro_sum

    cal_q = Calibration_Distribution(q, L_min, num_span)

    return cal_q


### compare two distributions
def compare_distribution(trainsets_x, trainsets_y, testsets_x, testsets_y, num_sample, num_repeat, num_span, L_min):
    loss_list = Train(trainsets_x, trainsets_y, testsets_x, testsets_y, num_sample, num_repeat)
    # print(loss_list)

    p = GetEPDF(loss_list, num_span)

    q = GetTPDF(L_min, num_sample, num_span, len(trainsets_x[0]) + 1)
    # print("Distribution P is", p)
    # print("Distribution Q is", q)

    return p, q


### main
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    output_config(args)

    # load data
    trainsets_x, trainsets_y, testsets_x, testsets_y = load_iris_train_test()

    L_min = calculate_min_loss(testsets_x, testsets_y)

    P_list = []
    Q_list = []
    for i in range(args.num_iter):
        num_sample = args.num_mini_sample * (i + 1)
        p, q = compare_distribution(trainsets_x, trainsets_y, testsets_x, testsets_y, num_sample, args.num_repeat,
                                    args.num_span, L_min)
        P_list.append(p)
        Q_list.append(q)

    path = '../user_data/tmp_data/'
    # parameters: number of repeated samples + number of intervals + minimal sample size + number of iterations
    paras = '({},{},{},{})'.format(args.num_repeat, args.num_span, args.num_mini_sample, args.num_iter)

    localtime = time.localtime(time.time())
    timestamp = "{:0>2d}{:0>2d}_{:0>2d}{:0>2d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                                       localtime.tm_min)
    utils.save_file(P_list, path + 'Iris_P{}({}).pkl'.format(paras, timestamp))
    utils.save_file(Q_list, path + 'Iris_Q{}({}).pkl'.format(paras, timestamp))


if __name__ == '__main__':
    print('start')
    T1 = time.time()

    # execute main function
    main()

    T2 = time.time()
    print('The runtime: %s ms' % ((T2 - T1) * 1000))

    print('end')
