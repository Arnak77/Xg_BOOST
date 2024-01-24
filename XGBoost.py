import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split


dataset=pd.read_csv(r"D:\NIT\JANUARY\11-12 JAN(enseble learn)\10th,11th\7.XGBOOST\Churn_Modelling.csv")
dataset.isnull().sum()

categorical_columns = dataset.select_dtypes(include=['object']).columns
categorical_columns

numerical_columns = dataset.select_dtypes(include=['number']).columns
numerical_columns

from sklearn.preprocessing import LabelEncoder
categorical_features = categorical_columns

for feature in categorical_features:
    le = LabelEncoder()
    dataset[feature] = le.fit_transform(dataset[feature])



X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.8}

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)



from xgboost import XGBClassifier
classifier1 = XGBClassifier(learning_rate=0.1,max_depth=4,n_estimators=200,subsample=0.8)
classifier1.fit(X_train, y_train)



y_pred = classifier1.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier1.score(X_train, y_train)
bias

variance = classifier1.score(X_test, y_test)
variance
















