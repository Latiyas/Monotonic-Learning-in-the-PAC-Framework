import os, math, random, time, utils
import numpy as np
import argparse


### parsing and configuration
def parse_args():
    desc = "Monotonicity analysis on the threshold function learning problem"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--num_scope', type=float, default=1, help='The scope of sampling')
    parser.add_argument('--goal', type=float, default=0.53, help='The ground truth of the threshold function')
    parser.add_argument('--num_repeat', type=int, default=1000, help='The number of repeated samplings')
    parser.add_argument('--num_span', type=int, default=100, help='The number of intervals')
    parser.add_argument('--num_mini_sample', type=int, default=200, help='The minimum sample size')
    parser.add_argument('--num_iter', type=int, default=50, help='The number of iterations')
    parser.add_argument('--tmp_dir', type=str, default='../user_data/tmp_data/',
                        help='Directory name to save temp file')

    return check_args(parser.parse_args())


### checking arguments
def check_args(args):
    # num_scope
    try:
        assert args.num_scope > 0
    except:
        print('The scope of sampling should be greater than 0')

    # goal
    try:
        assert args.goal >= 0 and args.goal <= args.num_scope
    except:
        print('The length of ground truth should be  between 0 and the scope of sampling')

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
    print('The ground truth: {}'.format(args.goal))
    print('The number of repeated samplings:{}'.format(args.num_repeat))
    print('The number of intervals:{}'.format(args.num_span))
    print('The minimum sample size:{}'.format(args.num_mini_sample))
    print('The number of iterations:{}'.format(args.num_iter))


### marking the instance
def mark_instance(goal, instance):
    # positive sample when x<a
    if instance < goal:
        label = 1
    else:
        label = 0
    return label


### generate the instance
def generate_instance(num_scope=1):
    # randomly choose a decimal between 0 and 1 (uniform distribution)
    instance = random.uniform(0, 1) * num_scope
    return instance


### generate the samples
def generate_samples(goal, num_scope, num_sample):
    instances = []
    labels = []
    for i in range(num_sample):
        instance = generate_instance(num_scope)
        instances.append(instance)
        labels.append(mark_instance(goal, instance))
    return instances, labels


### get the hypothesis
### use the smallest negative sample as a threshold
def get_hypothesis(num_scope, instances, labels):
    neg_min = num_scope
    for i in range(len(instances)):
        if labels[i] == 0 and instances[i] < neg_min:
            neg_min = instances[i]

    # print(neg_min)
    return neg_min


### loss function
### the loss is defined as the proportion of the distance between the target and the hypothesis over the entire interval under a uniform distribution
def loss_function(num_scope, goal, hyp):
    result = float(format((hyp - goal) / num_scope, '.4f'))

    return result


### training the model
def Train(goal, num_scope, num_sample, num_repeat=1):
    loss_list = []
    for i in range(num_repeat):
        instances, labels = generate_samples(goal, num_scope, num_sample)
        hyp = get_hypothesis(num_scope, instances, labels)
        test_loss = loss_function(num_scope, goal, hyp)
        loss_list.append(test_loss)

    return loss_list


### calculate the empirical probability density function - P
def GetEPDF(loss_list, num_span):
    # use np.histogram to get the distribution of frequencies
    p_num, _ = np.histogram(loss_list, bins=num_span, range=(0, 1))
    num_repeat = len(loss_list)
    p = [x / num_repeat for x in p_num]

    return p


### calculate the theoretical probability density function - Q (finite VC dimension)
def GetTPDF(num_sample, num_span, num_VC=1):
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

    return q


### compare two distributions
def compare_distribution(goal, num_scope, num_sample, num_repeat, num_span):
    loss_list = Train(goal, num_scope, num_sample, num_repeat)
    # print(loss_list)

    p = GetEPDF(loss_list, num_span)
    q = GetTPDF(num_sample, num_span, 1)
    # print('Distribution P is', p)
    # print('Distribution Q is', q)

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
        p, q = compare_distribution(args.goal, args.num_scope, num_sample, args.num_repeat, args.num_span)
        P_list.append(p)
        Q_list.append(q)

    path = args.tmp_dir
    paras = '({},{},{},{})'.format(args.num_repeat, args.num_span, args.num_mini_sample, args.num_iter)

    localtime = time.localtime(time.time())
    timestamp = "{:0>2d}{:0>2d}_{:0>2d}{:0>2d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                                       localtime.tm_min)
    utils.save_file(P_list, path + 'TH_P{}({}).pkl'.format(paras, timestamp))
    utils.save_file(Q_list, path + 'TH_Q{}({}).pkl'.format(paras, timestamp))


if __name__ == '__main__':
    print('start')
    T1 = time.time()

    # execute main function
    main()

    T2 = time.time()
    print('The runtime: %s ms' % ((T2 - T1) * 1000))

    print('end')
