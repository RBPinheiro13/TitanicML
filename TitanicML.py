import numpy as np
import pandas as pd
import sklearn
import os
from keras.models import Sequential
from keras.layers import Dense

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

useless_feat = ['Name', 'Ticket', 'Cabin', 'Embarked']

train_data = train_data.drop(columns=useless_feat)

# print(train_data.head())
# print(train_data.shape)
#
# print(train_data.isna().sum())
# print(test_data.isna().sum())

def rescale(train_data, test_data, column_name):

    max = np.max([np.max(train_data[column_name].loc[train_data[column_name].notnull()]),
                 np.max(test_data[column_name].loc[test_data[column_name].notnull()])])
    min = np.min([np.min(train_data[column_name].loc[train_data[column_name].notnull()]),
                 np.min(test_data[column_name].loc[test_data[column_name].notnull()])])


    train_data[column_name].loc[train_data[column_name].notnull()] = (train_data[column_name].loc[train_data[column_name].notnull()] - min)/(max - min)
    test_data[column_name].loc[test_data[column_name].notnull()] = (test_data[column_name].loc[test_data[column_name].notnull()] - min)/(max - min)


# Process age
rescale(train_data, test_data, "Age")

# Replace the nans
mean_age = np.mean(train_data["Age"])
print(mean_age)

values = {'Age': mean_age, 'Fare': -1}
train_data.fillna(value=values, inplace=True)
test_data.fillna(value=values, inplace=True)

# Rescale Fare
rescale(train_data, test_data, "Fare")

train_data['isAlone'] = np.bitwise_and(train_data['SibSp'] == 0, train_data['Parch'] == 0)
test_data['isAlone'] = np.bitwise_and(test_data['SibSp'] == 0, test_data['Parch'] == 0)

print(train_data.head())

#################################################################################
######################### Random Forest Classifier ##############################
#################################################################################
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "isAlone"]
X = pd.get_dummies(train_data[features], drop_first=True)
X_test = pd.get_dummies(test_data[features], drop_first=True)

model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=17)
model.fit(X, y)
success_rate = np.sum(model.predict(X) == y)/len(y)
print('Random Forest: {:.2f} %'.format(success_rate*100))

#################################################################################
############################## SVM Classifier ###################################
#################################################################################
from sklearn.svm import SVC

# Gaussian Kernel
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X, y)

success_rate = np.sum(svclassifier.predict(X) == y)/len(y)
print('SVM - Gaussian Kernel: {:.2f} %'.format(success_rate*100))

# Sigmoid Kernel
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X, y)

success_rate = np.sum(svclassifier.predict(X) == y)/len(y)
print('SVM - Sigmoid Kernel: {:.2f} %'.format(success_rate*100))

# Polynomial Kernel
for deg in np.arange(1,5):
    svclassifier = SVC(kernel='poly', degree=deg)
    svclassifier.fit(X, y)
    success_rate = np.sum(svclassifier.predict(X) == y)/len(y)
    print('SVM - Poly Kernel {} Degree: {:.2f} %'.format(deg, success_rate*100))


#################################################################################
######################### XGB Classifier ########################################
#################################################################################
from xgboost import XGBClassifier

model = XGBClassifier(max_depth=4)
model.fit(X, y)

success_rate = np.sum(model.predict(X) == y)/len(y)
print('XGB Classifier: {:.2f} %'.format(success_rate*100))

#################################################################################
######################### Keras Model ###########################################
#################################################################################
model = Sequential()
model.add(Dense(50, input_shape=X.shape, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=15)

success_rate = np.sum(model.predict(X) == y)/len(y)
print('Keras Model Classifier: {:.2f} %'.format(success_rate*100))


predictions = model.predict(X_test)
#
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('XGB.csv', index=False)
print("Your submission was successfully saved!")
