import sys
import pandas as pd
import numpy as np
from neural_network.nn import NeuralNetwork
from sklearn.preprocessing import MinMaxScaler
from preprocess import split


def Score(y_pred, y_true, weight):
    return np.sum(-weight * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))) / np.sum(weight)


if __name__ == '__main__':
    if len(sys.argv) < 5 or sys.argv[4] == '0':
        val = False
    else:
        val = True

    train_data = pd.read_csv(sys.argv[1])
    result = split.data_split(train_data, val=val, part=None)
    if val:
        x_train, x_valid, y_train, y_valid, weight = result
    else:
        x_train, y_train = result
        test_data = pd.read_csv(sys.argv[2])
        id_test = test_data['id']
        test_data = test_data.drop(['id', 'group1', 'code_id'], axis=1)
        x_valid = test_data

    Scaler = MinMaxScaler()
    Scaler.fit_transform(x_train)
    Scaler.transform(x_valid)
    clf = NeuralNetwork([x_train.shape[1], 128, 64, 32], 10)
    clf.fit(x_train, y_train)

    prob = clf.predict_proba(x_valid)[:, 0]
    if val:
        accuracy = clf.score(x_valid, y_valid)
        score = Score(prob, y_valid, weight)
        print(accuracy)
        print(score)
    else:
        result = {'id': id_test, 'proba': prob}
        result = pd.DataFrame(result)
        result.to_csv(sys.argv[3], index=False)
