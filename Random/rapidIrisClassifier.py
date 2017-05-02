#This is just to show how quickly you can cut together
#a classifier app with existing code bases
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

#load data
df = datasets.load_iris()

#Linear model
classifier = skflow.TensorFlowLinearClassifier(n_classes = 3)

#fit data
classifier.fit(df.data, df.target)

#Validate model, make predictions
score = metrics.accuracy_score(df.target, classifier.predict(df.data))

#Should be like 95%
print(score)