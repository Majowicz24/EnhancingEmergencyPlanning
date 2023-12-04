import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scikitplot as skplot
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from joblib import dump

scalar = StandardScaler()

os.chdir('D:\\RESEARCH')

df = pd.read_csv('D:\\RESEARCH\\metadata_sheets\\UpgradeZero.csv', low_memory=False)
df = df[df['in.geometry_building_type_height'] == 'Single-Family Detached']

df['Target_Variable'] =[
    1 if typ == 'Electricity' else 0 for typ in df['in.heating_fuel']
]
df.drop('in.heating_fuel', axis=1, inplace=True)

df = df.groupby('Target_Variable').apply(lambda x: x.sample(48720))

X = df[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]
Y = df[['Target_Variable']]
Y = Y.astype('int64')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=42, stratify= Y)
Y_train = Y_train['Target_Variable'].values
Y_test = Y_test['Target_Variable'].values

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

model = model = SVC(C = 10, random_state=42)

history = model.fit(X_train, Y_train)
predict = model.predict(X_test)

accuracy = accuracy_score(Y_test, predict)
precision = precision_score(Y_test, predict, average='micro')
recall = recall_score(Y_test, predict, average='micro')
f1 = f1_score(Y_test, predict, average='micro')
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)

cm = confusion_matrix(Y_test, predict)
skplot.metrics.plot_confusion_matrix(Y_test, predict)
plt.show()

dump(model, 'C:\\Users\\andre\\PYTHON\\svc_model.pkl')
dump(scalar, 'C:\\Users\\andre\\PYTHON\\data_scalar.pkl')
np.save('C:\\Users\\andre\\PYTHON\\X_test.npy', X_test)
np.save('C:\\Users\\andre\\PYTHON\\Y_test.npy', Y_test)

print(confusion_matrix(Y_test, predict))
print(classification_report(Y_test, predict))

