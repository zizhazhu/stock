import sys

if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv(sys.argv[1])
    dataset.drop('feature77', axis=1, inplace=True)
    dataset.drop('weight', axis=1, inplace=True)
    #dataset = dataset[dataset['group'] == 1]
    dataset.drop(['group', 'era'], axis=1, inplace=True)
    y = dataset['label'].astype('int')
    dataset.drop(['label', 'id'], axis=1, inplace=True)
    x = dataset
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import normalize
    print(x.describe())
    x = normalize(x)
    print(pd.DataFrame(x).describe())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
