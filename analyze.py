import sys
import pandas as pd


if __name__ == '__main__':
    dataset = pd.read_csv(sys.argv[1])
    print(dataset.info())
    print(dataset.describe())
    for i in range(1, 29):
        group_dataset = dataset[dataset['group'] == i]
        print("Group {} has {} entries".format(i, group_dataset.shape[0]))
        for j in range(1, 21):
            tmp_dataset = group_dataset[group_dataset['era'] == j]
            one = tmp_dataset[tmp_dataset['label'] == 1]
            zero = tmp_dataset[tmp_dataset['label'] == 0]
            print("Group:{} Era:{} 1:{} 0:{}".format(i, j, one.shape[0], zero.shape[0]))
