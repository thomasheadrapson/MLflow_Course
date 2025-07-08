# Imports librairies




from sklearn import svm, datasets 
from sklearn.model_selection import GridSearchCV


# from mlflow import MlflowClient
import mlflow 


# Set tracking experiment
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Define experiment name
apple_experiment = mlflow.set_experiment("Iris_Models")

# autolog runs
mlflow.autolog() 

iris = datasets.load_iris()

parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]} 

svc = svm.SVC()

clf = GridSearchCV(svc, parameters)

clf.fit(iris.data, iris.target)