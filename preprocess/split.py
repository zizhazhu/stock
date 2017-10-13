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


def data_split(dataset, val=True, part=None):
    # weight determine score
    if part is not None:
        dataset = dataset[dataset['group'] == part]
    else:
        dataset = one_hot_extend(dataset, 'group', remove=True)
    weight = dataset['weight']
    dataset = dataset.drop(['feature77', 'id', 'weight'], axis=1)
    y = dataset['label'].astype('int')
    dataset = dataset.drop('label', axis=1)
    x = dataset
    if val:
        train_idx = dataset['era'] != 20
        valid_idx = dataset['era'] == 20
        x = x.drop('era', axis=1)
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        weight = weight[valid_idx]
        return x_train, x_valid, y_train, y_valid, weight
    else:
        x = x.drop('era', axis=1)
        return x, y
