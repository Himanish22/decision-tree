import numpy as np
import pandas as pd

df = pd.read_csv('/path/to/updated_iris.csv')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
Y = df['FlowerType']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(classifier, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(Y_test, Y_pred)
print(confusion)
