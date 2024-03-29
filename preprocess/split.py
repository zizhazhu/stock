import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def one_hot_extend(dataset, label, remove=False):
    feature = dataset[label].values
    lb = LabelBinarizer()
    feature = pd.DataFrame(lb.fit_transform(feature))
    dataset = pd.concat([dataset, feature], axis=1)
    if remove:
        dataset = dataset.drop(label, axis=1)
    return dataset


def data_split(dataset, val=True, part=None, drop=None, era=20):
    # weight determine score
    if part:
        dataset = dataset[dataset['group1'] == part]
    weight = dataset['weight']
    dataset = dataset.drop('group1', axis=1)
    dataset = dataset.drop(['id', 'weight', 'code_id'], axis=1)
    if drop:
        dataset = dataset.drop([drop], axis=1)
    y = dataset['label'].astype('int')
    dataset = dataset.drop('label', axis=1)
    x = dataset

    # split train valid by era
    if val:
        train_idx = dataset['era'] != era
        valid_idx = dataset['era'] == era
        x = x.drop('era', axis=1)
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        weight = weight[valid_idx]
        return x_train, x_valid, y_train, y_valid, weight
    else:
        x = x.drop('era', axis=1)
        return x, y
