import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier

import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.metrics import f1_score, confusion_matrix ,accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 


dataset = pd.read_csv('KaggleV2-May-2016.csv')



dataset['No-show'] = dataset['No-show'].apply(lambda x: 0 if x.strip()=='No' else 1)

missed_appointment = dataset.groupby('PatientId')['No-show'].sum()

missed_appointment = missed_appointment.to_dict()

dataset['missed_appointment_before'] = dataset.PatientId.map(lambda x: 1 if missed_appointment[x]>0 else 0)
print(dataset['missed_appointment_before'].corr(dataset['No-show']))

dataset = dataset.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis = 1)
print("Columns: {}".format(dataset.columns))



dataset = pd.concat([dataset.drop('Neighbourhood', axis = 1), 
           pd.get_dummies(dataset['Neighbourhood'])], axis=1)


dataset['Gender'] = dataset['Gender'].map({'M': 0, 'F': 1})


y = dataset.loc[:, 'No-show']
X = dataset.drop(['No-show'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

print("Final shape: {}".format(X_train.shape))


standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)


classifier = Sequential()
classifier.add(Dense(units = 64, activation = 'relu', input_dim = 90))
classifier.add(Dropout(rate = 0.5))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(rate = 0.5))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(rate = 0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()


history = classifier.fit(X_train, y_train, epochs = 10, validation_split = 0.2)



y_preds = classifier.predict(X_test)
y_preds = y_preds > 0.5
y_preds = [int(i) for i in y_preds]
print('NN - Accuracy: {:2.2f}%'.format(accuracy_score(y_test, y_preds) * 100))
print('NN - Precision score: {:2.2f}%'.format(precision_score(y_test, y_preds)*100))
print('NN - Recall score: {:2.2f}%'.format(recall_score(y_test, y_preds)*100))
print('NN - F1-score: {:2.2f}%'.format(f1_score(y_test, y_preds) * 100))

NN = history.history['accuracy']


clf1 = RandomForestClassifier(random_state = 0)
clf1.fit(X_train, y_train)


y_preds = clf1.predict(X_test)
print('RF - Accuracy: {:2.2f}%'.format(accuracy_score(y_test, y_preds) * 100))
print('RF - Precision score: {:2.2f}%'.format(precision_score(y_test, y_preds)*100))
print('RF - Recall score: {:2.2f}%'.format(recall_score(y_test, y_preds)*100))
print('RF - F1-score: {:2.2f}%'.format(f1_score(y_test, y_preds) * 100))

print(confusion_matrix(y_test, y_preds))

RF = cross_val_score(clf1,X_train,y_train,cv = 10)


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_preds = gbc.predict(X_test)
print('GB - Accuracy: {:2.2f}%'.format(accuracy_score(y_test, y_preds) * 100))
print('GB - Precision score: {:2.2f}%'.format(precision_score(y_test, y_preds)*100))
print('GB - Recall score: {:2.2f}%'.format(recall_score(y_test, y_preds)*100))
print('GB - F1-score: {:2.2f}%'.format(f1_score(y_test, y_preds) * 100))

print(confusion_matrix(y_test, y_preds))

GB = cross_val_score(gbc,X_train,y_train,cv = 10)

from scipy.stats import ttest_ind

ttest,pval = ttest_ind(NN,list(RF))
if pval <0.05:
  print("we reject the null hypothesis that assumes that the two samples have the same distribution.")
else:
  print("we accept null hypothesis, are in different distributions")

ttest,pval = ttest_ind(list(RF),list(GB))
if pval <0.05:
  print("we reject the null hypothesis that assumes that the two samples have the same distribution.")
else:
  print("we accept null hypothesis, are in different distributions")

ttest,pval = ttest_ind(NN,list(GB))
if pval <0.05:
  print("we reject the null hypothesis that assumes that the two samples have the same distribution.")
else:
  print("we accept null hypothesis, are in different distributions")

imp = pd.DataFrame(clf1.feature_importances_, X.columns)
imp = imp.sort_values(by=0, ascending=False)
print(imp.head(10))
plt.plot(imp.head())
plt.show()


imp = pd.DataFrame(gbc.feature_importances_, X.columns)
imp = imp.sort_values(by=0, ascending=False)
print(imp.head(10))
plt.plot(imp.head())
plt.show()
