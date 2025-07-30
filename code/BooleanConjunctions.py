import os, math, random, time, utils
import numpy as np
import argparse


### parsing and configuration
def parse_args():
    desc = "Monotonicity analysis on the Boolean literal conjunction learning problem"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--goal', type=int, nargs='*', default=[0, 1, 2, 2, 2, 0, 0, 2, 1, 2],
                        help='The ground truth of the Boolean literal conjunction')
    parser.add_argument('--num_repeat', type=int, default=1000, help='The number of repeated samplings')
    parser.add_argument('--num_span', type=int, default=100, help='The number of intervals')
    parser.add_argument('--num_mini_sample', type=int, default=25, help='The minimum sample size')
    parser.add_argument('--num_iter', type=int, default=50, help='The number of iterations')
    parser.add_argument('--tmp_dir', type=str, default='../user_data/tmp_data/',
                        help='Directory name to save temp file')

    return check_args(parser.parse_args())


### checking arguments
def check_args(args):
    # goal
    try:
        len_goal = len(args.goal)
        assert len_goal >= 1
    except:
        print('The length of ground truth should be no less than 1')

    # set args.num_byte to the length of args.goal
    setattr(args, 'num_byte', len(args.goal))

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
    print('The ground truth: [{}]'.format(str(args.goal)[1:-1]))
    print('The number of repeated samplings:{}'.format(args.num_repeat))
    print('The number of intervals:{}'.format(args.num_span))
    print('The minimum sample size:{}'.format(args.num_mini_sample))
    print('The number of iterations:{}'.format(args.num_iter))


### marking the instance
def mark_instance(goal, instance):
    # summing by bits is 1 for negative samples, i.e., the target and sample are 0 and 1, respectively.
    if (1 in [i + j for i, j in zip(goal, instance)]):
        label = 0
    else:
        label = 1

    return label


### generate the instance
def generate_instance(num_byte, isGoal=2):
    instance = []
    for i in range(num_byte):
        instance.append(random.randrange(isGoal))

    return instance


### generate the samples
def generate_samples(goal, num_byte, num_sample):
    instances = []
    labels = []
    for i in range(num_sample):
        instance = generate_instance(num_byte)
        instances.append(instance)
        labels.append(mark_instance(goal, instance))

    return instances, labels


### get the hypothesis
def get_hypothesis(instances, labels):
    if len(instances) == 0:
        return []

    num_byte = len(instances[0])
    pos_examples = np.array([0] * num_byte)
    num_pos = 0
    for i, j in zip(instances, labels):
        if j == 1:
            num_pos += 1
            pos_examples = pos_examples + np.array(i)

    if num_pos == 0:
        return [2] * num_byte

    hyp = []
    for i in pos_examples:
        if i == 0:
            hyp.append(0)
        elif i == num_pos:
            hyp.append(1)
        else:
            hyp.append(2)

    return hyp


### loss function
### the loss is 0 if all booleans can be correctly identified or the hypothesis consists of all '?'
def loss_function(goal, hyp):
    num_byte = len(goal)
    inde_mis = 0
    inde_true = 0
    for i, j in zip(goal, hyp):
        if i != j:
            inde_mis += 1
        else:
            if i == 2:
                inde_true += 1

    if inde_mis == 0:
        tmp = 0
    elif inde_mis + inde_true == num_byte:
        tmp = (2 ** inde_true) / (2 ** num_byte)
    else:
        tmp = (2 ** inde_mis - 1) / (2 ** (num_byte - inde_true))

    result = float(format(tmp, '.4f'))

    return result


### training the model
def Train(goal, num_byte, num_sample, num_repeat=1):
    loss_list = []
    for i in range(num_repeat):
        instances, labels = generate_samples(goal, num_byte, num_sample)
        hyp = get_hypothesis(instances, labels)
        test_loss = loss_function(goal, hyp)
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


### calculate the theoretical probability density function - Q (finite hypothesis space)
def GetTPDF(num_sample, num_span, num_byte):
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


### compare two distributions
def compare_distribution(goal, num_byte, num_sample, num_repeat, num_span):
    loss_list = Train(goal, num_byte, num_sample, num_repeat)
    # print(loss_list)

    p = GetEPDF(loss_list, num_span)
    q = GetTPDF(num_sample, num_span, num_byte)
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

    P_list = []
    Q_list = []
    for i in range(args.num_iter):
        num_sample = args.num_mini_sample * (i + 1)
        p, q = compare_distribution(args.goal, args.num_byte, num_sample, args.num_repeat, args.num_span)
        P_list.append(p)
        Q_list.append(q)

    path = args.tmp_dir
    # parameters: number of repeated samplings + number of intervals + minimal sample size + number of iterations
    paras = '({},{},{},{})'.format(args.num_repeat, args.num_span, args.num_mini_sample, args.num_iter)

    localtime = time.localtime(time.time())
    timestamp = "{:0>2d}{:0>2d}_{:0>2d}{:0>2d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                                       localtime.tm_min)
    utils.save_file(P_list, path + 'CBL_P{}({}).pkl'.format(paras, timestamp))
    utils.save_file(Q_list, path + 'CBL_Q{}({}).pkl'.format(paras, timestamp))


if __name__ == '__main__':
    print('start')
    T1 = time.time()

    # execute main function
    main()

    T2 = time.time()
    print('The runtime: %s ms' % ((T2 - T1) * 1000))

    print('end')
