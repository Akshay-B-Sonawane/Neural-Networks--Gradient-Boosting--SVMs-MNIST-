
from sklearn.datasets import fetch_openml
# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

from sklearn import ensemble
from sklearn.kernel_approximation import Nystroem
Estimators = [50,70]
Learning_rates = [0.1,0.15,0.20]
Max_depths = [1,3]
#Leafs = [1]

#feature_map_nystroem = Nystroem()
#data_transformed = feature_map_nystroem.fit_transform(X)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

for estimate in Estimators:
  for lrates in Learning_rates:
    for max_depth in Max_depths:
        clf = ensemble.GradientBoostingClassifier(n_estimators = estimate, learning_rate = lrates, max_depth = max_depth)
        clf.fit(X_train, y_train)
        print("ERROR RATE FOR No_Estimators: "+str(estimate)+", Learning rate: "+str(lrates)+", Maximum depth: "+str(max_depth)+" is")
        y_pred = clf.predict(X_test)
        from sklearn import metrics
        print((1-metrics.accuracy_score(y_test, y_pred))*100)