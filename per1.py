from sklearn.neural_network import MLPClassifier

Layers = [(100,) , (10,2) , (5,2) , (50,30)]
alphas = [0.0001, 0.001, 0.01, 0.1]
activations = ['tanh', 'relu']

for layer in Layers:
  for alpha in alphas:
    for active in activations:

      clf = MLPClassifier(alpha=alpha, hidden_layer_sizes=layer, activation=active)
      clf.fit(X_train, y_train) 
      y_pred = clf.predict(X_test)
      print("ERROR RATE for Layer: "+str(layer)+" , alpha: "+str(alpha)+" , activations: "+active+" is")
      from sklearn import metrics
      print(1-metrics.accuracy_score(y_test, y_pred))