import pandas as pd


def data_split(dataset, val=True, part=False):
    # weight determine score
    if part:
        dataset = dataset[dataset['group'] == 1]
    dataset = dataset.drop(['feature77', 'weight', 'id', 'group'], axis=1)
    y = dataset['label'].astype('int')
    dataset = dataset.drop('label', axis=1)
    x = dataset
    if val:
        train_idx = dataset['era'] != 20
        valid_idx = dataset['era'] == 20
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        return x_train, x_valid, y_train, y_valid
    else:
        return x, y
