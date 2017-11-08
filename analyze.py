import sys
import pandas as pd


if __name__ == '__main__':
    dataset = pd.read_csv(sys.argv[1])
    print(dataset.info())
    print(dataset.describe())
    data_group = {}
    for data_index, data_line in dataset.iterrows():
        code_id = int(data_line['code_id'])
        group2 = int(data_line['group2'])
        if code_id in data_group:
            if data_group[code_id] != group2:
                print("not")
                break
        else:
            data_group[code_id] = group2
    else:
        print("yes")
