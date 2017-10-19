import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from preprocess import split


def Score(y_pred, y_true, weight):
    return np.sum(-weight * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))) / np.sum(weight)


if __name__ == '__main__':
    val = False

    train_data = pd.read_csv(sys.argv[1])
    result = split.data_split(train_data, val=val, part=None)
    if val:
        x_train, x_valid, y_train, y_valid, weight = result
    else:
        x_train, y_train = result
        test_data = pd.read_csv(sys.argv[2])
        id_test = test_data['id']
        test_data = split.one_hot_extend(test_data, 'group', remove=True)
        test_data = test_data.drop(['id', 'feature77'], axis=1)
        test_data = test_data.drop(['id', 'feature43'], axis=1)
        x_valid = test_data

    clf = RandomForestClassifier(n_estimators=200, max_depth=6, oob_score=True, n_jobs=-3)
    clf.fit(x_train, y_train)

    prob = clf.predict_proba(x_valid)[:, 1]
    if val:
        accuracy = clf.score(x_valid, y_valid)
        score = Score(prob, y_valid, weight)
        print(accuracy)
        print(score)
    else:
        result = {'id': id_test, 'proba': prob}
        result = pd.DataFrame(result)
        result.to_csv(sys.argv[3], index=False)
