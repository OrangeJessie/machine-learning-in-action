from numpy import*
import operator
import matplotlib
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


def auto_norm(data_set):
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    range = max_value - min_value


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

train_group, train_labels = create_data()
real_type = classify([0, 0], train_group, train_labels, 3)
train_group2, train_labels2 = file2matrix('datingTestSet.txt')
train_labels3 = str2int(train_labels2)
real_type2 = classify([20000, 12, 1], train_group2, train_labels2, 10)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(train_group2[:, 0], train_group2[:, 1], 15*array(train_labels3), 15*array(train_labels3))
plt.show()
