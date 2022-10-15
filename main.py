import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from LogisticModelTree import LogisticModelTree
import sklearn.metrics
from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt


def analyze_data(data):

    print(data.isna().sum())
    print(data.describe())

    print(data[data.columns[1:]].corr()['target'][:])


def preprocess(data):
    int_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for i in int_cols:
        data[i].fillna(value=data[i].mean(), inplace=True)
    for i in cat_cols:
        data[i].fillna(value=data[i].mode()[0], inplace=True)
    y = data['target']
    X = data.drop(columns=['target'])
    scaler = StandardScaler()
    cols=X.columns
    X = scaler.fit_transform(X)
    X=pd.DataFrame(X,columns=cols)
    return X, y

def plot_roc_auc_curve(max_depth=5, min_leaf=5):
    data = pd.read_csv('heart.csv')
    analyze_data(data)
    x, y = preprocess(data)

    tree = LogisticModelTree()

    k = 10
    cv = KFold(n_splits=k, random_state=1, shuffle=True)

    arr_fpr, arr_tpr = [], []

    for train_index, test_index in cv.split(x):
        train_X = x.iloc[train_index]
        test_X = x.iloc[test_index]
        train_y = y.iloc[train_index]
        test_y = y.iloc[test_index]

        tree.fit(train_X, train_y, max_depth=max_depth, min_leaf=min_leaf)
        y_true, y_pred = test_y, tree.predict(test_X)
        fpr, tpr, thresh = roc_curve(y_true, y_pred)
        arr_fpr.append(fpr)
        arr_tpr.append(tpr)

    fpr = sum(arr_fpr) / len(arr_fpr)
    tpr = sum(arr_tpr) / len(arr_tpr)
    auc = sklearn.metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %.3f)' % auc)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.legend(loc='lower right')
    plt.title('Receiver operating characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


if __name__ == "__main__":
    plot_roc_auc_curve(2,10)
    plot_roc_auc_curve(10,2)
    plot_roc_auc_curve()




