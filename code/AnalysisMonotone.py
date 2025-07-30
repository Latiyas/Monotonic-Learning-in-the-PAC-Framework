import os, math, random, time, utils
import argparse


### parsing and configuration
def parse_args():
    desc = "Monotonicity analysis on the Boolean literal conjunction learning problem"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--num_repeat', type=int, default=1000, help='The number of repeated sampling')
    parser.add_argument('--num_span', type=int, default=100, help='The number of intervals')
    parser.add_argument('--num_mini_sample', type=int, default=25, help='The minimum sample size')
    parser.add_argument('--num_iter', type=int, default=50, help='The number of iterations')

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

    return args


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
    def __init__(self, data_name, num_mini_sample, num_iter=50, num_repeat=100, num_span=100):
        self.data_name = data_name
        self.num_mini_sample = num_mini_sample
        self.num_iter = num_iter
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


### main
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    CBL = Dataset('CBL', num_mini_sample=args.num_mini_sample, num_iter=args.num_iter, num_repeat=args.num_repeat,
                  num_span=args.num_span)

    files = utils.find_files(CBL.path,
                             '({},{},{},{})'.format(CBL.num_repeat, CBL.num_span, CBL.num_mini_sample, CBL.num_iter))
    if len(files) > 0:
        CBL_name = files[0]
    else:
        print("Warning: There is no file to be matched.")
        exit()

    CBL.analyze_monotone(CBL_name)


if __name__ == '__main__':
    print('start')

    # execute main function
    main()

    print('end')
