# find out who to date with
from numpy import*
import operator
import matplotlib.pyplot as plt


def create_data():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify(inx, data_set, labels, k):         # calculate distance between train set and unknown set
    data_set_size = data_set.shape[0]
    diff_data = tile(inx, (data_set_size, 1)) - data_set
    pow_data = diff_data ** 2
    sum_data = pow_data.sum(axis=1)
    distance = sum_data ** 0.5
    sorted_data = distance.argsort()
    data_label_dict = {}
    for num in range(k):
        label = labels[sorted_data[num]]
        data_label_dict[label] = data_label_dict.get("label", 0) + 1
    sorted_dict = sorted(data_label_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_dict[0][0]


def file2matrix(filename):                      # change file into matrix
    fr = open(filename)
    array_lines = fr.readlines()
    num_of_lines = len(array_lines)
    train_list = zeros((num_of_lines, 3))
    class_label = []
    index = 0
    for line in array_lines:
        line = line.strip()
        list_from_line = line.split("\t")
        train_list[index, :] = list_from_line[0:3]
        class_label.append(list_from_line[3])
        index += 1
    return train_list, class_label


def auto_norm(data_set):                    # normalize data
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    range_of_data = max_value - min_value
    len_of_data = data_set.shape[0]
    data_set_out = data_set - tile(min_value, (len_of_data, 1))
    data_set_out = data_set_out/tile(range_of_data, (len_of_data, 1))
    return data_set_out, range_of_data, min_value


def str2int(str_list):                  # change string label into int label
    int_str = []
    for component in str_list:
        if component == "largeDoses":
            int_str.append(3)
        elif component == "smallDoses":
            int_str.append(2)
        else:
            int_str.append(1)
    return int_str


def data_test():                    # use some data to test the algorithm
    ratio = 0.1
    data, label = file2matrix('datingTestSet.txt')
    label = str2int(label)
    data_set, ranges, min_data = auto_norm(data)
    num_data = data_set.shape[0]
    num_test = int(ratio*num_data)
    num_error = 0.0
    for i in range(num_test):
        result = classify(data_set[i, :], data_set[num_test:num_data, :], label[num_test:num_data], 10)
        print('the classify result is: {}, the real data is: {}'.format(result, label[i]))
        if result != label[i]:
            num_error += 1.0
    print('the total error rate is: {}'.format(num_error/float(num_test)))


def classify_people():
    people_data1 = float(input('input percentage of time spend playing video games:'))
    people_data2 = float(input('input frequent flier miles per year:'))
    people_data3 = float(input('input eat ice cream per year:'))
    dating_data, dating_label = file2matrix('datingTestSet.txt')
    norm_data, data_range, data_min = auto_norm(dating_data)
    dating_label = str2int(dating_label)
    new_people = array([people_data2, people_data1, people_data3])
    kind_of_people = int(classify((new_people - data_min)/data_range, norm_data, dating_label, 3))
    label_list = ['no interest', 'smallDoses', 'largeDoses']
    print(label_list[kind_of_people - 1])
    return label_list[kind_of_people - 1]


def plot_fig(train_group, train_labels):                    # plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_group[:, 0], train_group[:, 1], 15*array(train_labels), 15*array(train_labels))
    plt.show()


data_test()
classify_people()
