import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    val = False

    train_data = pd.read_csv(sys.argv[1])
    train_data.drop(['feature77', 'weight', 'era', 'id'], axis=1, inplace=True)
    # train_data = train_data[train_data['group'] == 1]
    train_data.drop('group', axis=1, inplace=True)
    y = train_data['label'].astype('int')
    train_data.drop('label', axis=1, inplace=True)
    x = train_data

    # x = normalize(x)
    if val:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    else:
        x_train, y_train = x, y
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=2)
    clf.fit(x_train, y_train)

    test_data = pd.read_csv(sys.argv[2])
    id_test = test_data['id']
    test_data.drop(['id', 'group', 'feature77'], axis=1, inplace=True)
    x_test = test_data
    y_test = clf.predict_proba(x_test)[:, 1]
    result = {'id': id_test, 'proba': y_test}
    result = pd.DataFrame(result)
    result.to_csv(sys.argv[3], index=False)
