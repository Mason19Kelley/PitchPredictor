
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)

## Importing the dataset

atbats = pd.read_csv('2019_atbats.csv')
pitches = pd.read_csv('2019_pitches.csv')
pitch = pd.merge(pitches, atbats, on="ab_id",how='left')

pitch = pitch.dropna(subset=["pitch_type"])
pitch = pitch[pitch.g_id < 201901001]

X = pitch.iloc[:, [33,34,37,38,39,46,47]].values
y = pitch.iloc[:, 29].values
X3 = np.zeros(y.shape)

X3 = np.where(X3==0,'FF', 'FF')

X3[0] = 'FF'
X3[1:] = pitch.iloc[:-1, 29].values

# X3 = np.where(X3 == 'FF', 'FB', 'OS')
# y = np.where(y == 'FF', 'FB', 'OS')


X3 = np.where(X3 == 'FF', 'FB', X3)
X3 = np.where(X3 == 'FT', 'FB', X3)
X3 = np.where(X3 == 'SI', 'FB', X3)
X3 = np.where(X3 == 'FO', 'FB', X3)

X3 = np.where(X3 == 'SL', 'OS', X3)
X3 = np.where(X3 == 'FC', 'OS', X3)
X3 = np.where(X3 == 'KC', 'OS', X3)
X3 = np.where(X3 == 'CU', 'OS', X3)

X3 = np.where(X3 == 'FS', 'CH', X3)
X3 = np.where(X3 == 'CH', 'CH', X3)
X3 = np.where(X3 == 'KN', 'CH', X3)


y = np.where(y == 'FF', 'FB', y)
y = np.where(y == 'FT', 'FB', y)
y = np.where(y == 'FS', 'FB', y)
y = np.where(y == 'FO', 'FB', y)

y = np.where(y == 'SL', 'OS', y)
y = np.where(y == 'FC', 'OS', y)
y = np.where(y == 'KC', 'OS', y)
y =  np.where(y == 'CU', 'OS', y)

y = np.where(y == 'FS', 'CH', y)
y = np.where(y == 'CH', 'CH', y)
y = np.where(y == 'KN', 'CH', y)





ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5,6])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X3 = X3.reshape(-1,1)

print(X3.shape)


print(X)

X = np.concatenate((X, X3), axis=1)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

## Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

clf2 = SVC(kernel='rbf', C=10, gamma=.1)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)

print(accuracy_score(y_test, y_pred))

n = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
np.savetxt('preds.csv', n, delimiter=',',fmt="%s")

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01],'kernel': ['rbf']}
# grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
# grid.fit(X_train,y_train)
# print(grid.best_estimator_)
# best_accuracy = grid.best_score_
# best_parameters = grid.best_params_

# print(best_accuracy)
# print(best_params_)