import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
final= pd.read_csv('gender_submission.csv')

dataset_train=dataset_train.fillna(dataset_train.mean())
y_test= final.iloc[:,1]
final=final.iloc[:,0].values

# Create a set of dummy variables from the Embarked variable
X_train_e = pd.get_dummies(dataset_train["Embarked"])
# Join the dummy variables to the main dataframe
X_train_e = X_train_e.iloc[:, 0:-1]
dataset_train = pd.concat([dataset_train, X_train_e], axis=1)

X_test_e = pd.get_dummies(dataset_test["Embarked"])
X_test_e = X_test_e.iloc[:, 0:-1]
dataset_test = pd.concat([dataset_test, X_test_e], axis=1)

X_train_sex = pd.get_dummies(dataset_train["Sex"])
# Join the dummy variables to the main dataframe
X_train_sex = X_train_sex.iloc[:, 0:-1]
dataset_train = pd.concat([dataset_train, X_train_sex], axis=1)

X_test_sex = pd.get_dummies(dataset_test["Sex"])
X_test_sex = X_test_sex.iloc[:, 0:-1]
dataset_test = pd.concat([dataset_test, X_test_sex], axis=1)

dataset_train['Cabin'] = dataset_train['Cabin'].apply(lambda x:0 if str(x)=='nan' else ord(x[0])-ord('A')+1)
dataset_test['Cabin'] = dataset_test['Cabin'].apply(lambda x:0 if str(x)=='nan' else ord(x[0])-ord('A')+1)

dataset_train['Age'] = dataset_train['Cabin'].apply(lambda x:0 if x<18 else 1)
dataset_test['Age'] = dataset_test['Cabin'].apply(lambda x:0 if x<18 else 1)

cols = list(dataset_train.columns.values)
cols.pop(cols.index('Survived'))
cols.pop(cols.index('PassengerId'))
cols.pop(cols.index('Sex'))
cols.pop(cols.index('Embarked'))
cols.pop(cols.index('Name'))
cols.pop(cols.index('Ticket'))
#cols.pop(cols.index('Cabin'))
cols.pop(cols.index('Age'))
cols.pop(cols.index('Fare'))
 #Remove b from list
dataset_train = dataset_train[cols+['Survived']]
X_train = dataset_train.iloc[:, 0:-1]
y_train = dataset_train.iloc[:, -1]

cols = list(dataset_test.columns.values)

cols.pop(cols.index('Sex'))
cols.pop(cols.index('Embarked'))
cols.pop(cols.index('PassengerId'))
cols.pop(cols.index('Name'))
cols.pop(cols.index('Ticket'))
#cols.pop(cols.index('Cabin'))
cols.pop(cols.index('Age'))
cols.pop(cols.index('Fare'))
 #Remove b from list
dataset_test = dataset_test[cols]
X_test=dataset_test.iloc[:, 0:]


X_train = X_train.as_matrix().astype(np.float)
y_train = y_train.as_matrix().astype(np.float)
X_test = X_test.as_matrix().astype(np.float)


X_test=np.nan_to_num(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


final=pd.concat([pd.DataFrame(final),pd.DataFrame(y_pred)], axis=1)
final.columns = ['PassengerId', 'Survived']
final.to_csv('result.csv', sep=',')


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm