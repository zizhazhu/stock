import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from preprocess import split

if __name__ == '__main__':
    val = True

    train_data = pd.read_csv(sys.argv[1])
    result = split.data_split(train_data, val=val)
    if val:
        x_train, x_valid, y_train, y_valid, weight_train, weight_valid = result
    else:
        x_train, y_train, weight = result
        test_data = pd.read_csv(sys.argv[2])
        id_test = test_data['id']
        test_data.drop(['id', 'group', 'feature77'], axis=1, inplace=True)
        x_valid = test_data

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=2, class_weight=weight_train)
    clf.fit(x_train, y_train)

    if val:
        score = clf.score(x_valid, y_valid)
        print(score)
    else:
        y_test = clf.predict_proba(x_valid)[:, 1]
        result = {'id': id_test, 'proba': y_test}
        result = pd.DataFrame(result)
        result.to_csv(sys.argv[3], index=False)
