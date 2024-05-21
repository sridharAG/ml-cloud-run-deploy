from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report   
import joblib 

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaling', MinMaxScaler()), 
    ('clf', DecisionTreeClassifier(max_depth=2))
    ])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))

print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("Classification Report: ")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

joblib.dump(pipeline, 'model.joblib')
print("Model saved as model.joblib")

