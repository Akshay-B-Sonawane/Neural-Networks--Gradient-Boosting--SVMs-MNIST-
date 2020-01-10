from sklearn.datasets import fetch_openml
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import metrics
# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]








Kernels = ['rbf', 'linear', 'sigmoid']
#Penalties = ['l2', 'l1', 'none']
C = [0.01, 0.1, 0.11, 1]

for kernel in Kernels:
  for penalty in C:

    clf = SVC(C = penalty, kernel = kernel)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("ERROR RATE of Kernel: "+kernel+" and penalty: "+str(penalty)+"is " )
    print((1-metrics.accuracy_score(y_test, y_pred))*100)