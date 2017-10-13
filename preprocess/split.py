import pandas as pd


def data_split(dataset, val=True, part=None):
    # weight determine score
    if part is not None:
        dataset = dataset[dataset['group'] == part]
    weight = dataset['weight']
    dataset = dataset.drop(['feature77', 'id', 'weight', 'group'], axis=1)
    y = dataset['label'].astype('int')
    dataset = dataset.drop('label', axis=1)
    x = dataset
    if val:
        train_idx = dataset['era'] != 20
        valid_idx = dataset['era'] == 20
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        weight_train, weight_valid = weight[train_idx], weight[valid_idx]
        return x_train, x_valid, y_train, y_valid, weight_train, weight_valid
    else:
        return x, y, weight
