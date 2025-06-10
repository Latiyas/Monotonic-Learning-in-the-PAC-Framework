import os, math, random, time, utils
import numpy as np


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
            tmp = 1 - 4 * H * math.exp(-1 * num_sample * (x**2) / 32)
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


if __name__ == '__main__':
    print('start')

    T1 = time.time()

    # assuming the problem is between 0 and 1
    num_scope = 1

    # generate a target threshold (assumed to be a decimal number between 0 and 1), the distribution is homogeneous
    # example
    # goal = generate_instance(num_scope)
    goal = 0.53
    # print(goal)

    # number of repeated samples
    num_repeat = 1000
    # number of intervals
    num_span = 100
    # minimal sample size
    num_mini_sample = 200
    # number of iterations
    num_round = 50
    P_list = []
    Q_list = []
    for i in range(num_round):
        num_sample = num_mini_sample * (i + 1)
        p, q = compare_distribution(goal, num_scope, num_sample, num_repeat, num_span)
        P_list.append(p)
        Q_list.append(q)

    path = '../user_data/tmp_data/'
    paras = '({},{},{},{})'.format(num_repeat, num_span, num_mini_sample, num_round)

    localtime = time.localtime(time.time())
    timestamp = "{:0>2d}{:0>2d}_{:0>2d}{:0>2d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    utils.save_file(P_list, path + 'TH_P{}({}).pkl'.format(paras, timestamp))
    utils.save_file(Q_list, path + 'TH_Q{}({}).pkl'.format(paras, timestamp))

    T2 = time.time()
    print('The runtime: %s ms' % ((T2 - T1) * 1000))

    print('end')
