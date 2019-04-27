import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from matplotlib import pyplot as plt
from datetime import date
from sklearn.model_selection import `_test_split as tts
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression as LR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor as GBR

train = pd.read_csv('E:/Incendo/train_file.csv')
test = pd.read_csv('E:/Incendo/test_file.csv')

for col in train.columns:
    print(col, train[col].nunique())
    
train.drop(['Description','Greater_Risk_Question','GeoLocation'], axis=1, inplace=True)

# Categorical data
race = pd.get_dummies(train['Race']).iloc[:,1:]
race['Other'] = race['Total']
race.drop('Total', axis=1, inplace=True)
qcode = pd.get_dummies(train['QuestionCode']).iloc[:,1:]
sid1 = pd.get_dummies(train['StratID1']).iloc[:,1:]
sid1.columns = ['sid1_1','sid1_2']
sid2 = pd.get_dummies(train['StratID2']).iloc[:,1:]
sid2.columns = ['sid2_1','sid2_2','sid2_3','sid2_4','sid2_5','sid2_6','sid2_7']
sid3 = pd.get_dummies(train['StratID3']).iloc[:,1:]
sid3.columns = ['sid3_1','sid3_2','sid3_3','sid3_4']
year = pd.get_dummies(train['YEAR']).iloc[:,1:]
sex = pd.get_dummies(train['Sex']).iloc[:, 1:]
sex.columns=['Male','Trans']
loc = pd.get_dummies(train['LocationDesc']).iloc[:, 1:]
stype = pd.get_dummies(train['StratificationType']).iloc[:, 1:]
stype.columns=['National', 'other', 'State', 'Territory']
size = pd.cut(train['Sample_Size'], 1000, labels=False)
grade = pd.get_dummies(train['Grade']).iloc[:, 1:]

X = train.iloc[:, [3, 4,7]]
y = train.iloc[:,13].values
X = pd.concat([X, race, qcode, sid1, sid2, sid3, year, sex, loc, stype, grade, size], axis=1)


#X['Grade'] = X['Grade'].astype('category')
#X['Grade'] = X['Grade'].cat.reorder_categories([0,1,2,3,4], ordered=True)
#X['Grade'] = X['Grade'].cat.codes
#samplesize = X['Sample_Size'].values

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)
df = X.head(10)
for col in df.columns:
    print(col, df[col].dtype)


dectree.fit(X_train, y_train)
y_pred = dectree.predict(X_test)

forest.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))

# On testdata
test.drop(['Description','Greater_Risk_Question','GeoLocation'], axis=1, inplace=True)

race = pd.get_dummies(test['Race']).iloc[:,1:]
race['Other'] = race['Total']
race.drop('Total', axis=1, inplace=True)
qcode = pd.get_dummies(test['QuestionCode']).iloc[:,1:]
sid1 = pd.get_dummies(test['StratID1']).iloc[:,1:]
sid1.columns = ['sid1_1','sid1_2']
sid2 = pd.get_dummies(test['StratID2']).iloc[:,1:]
sid2.columns = ['sid2_1','sid2_2','sid2_3','sid2_4','sid2_5','sid2_6','sid2_7']
sid3 = pd.get_dummies(test['StratID3']).iloc[:,1:]
sid3.columns = ['sid3_1','sid3_2','sid3_3','sid3_4']
year = pd.get_dummies(test['YEAR']).iloc[:,1:]
sex = pd.get_dummies(test['Sex']).iloc[:, 1:]
sex.columns=['Male','Trans']
loc = pd.get_dummies(test['LocationDesc']).iloc[:, 1:]
stype = pd.get_dummies(test['StratificationType']).iloc[:, 1:]
stype.columns=['National', 'other', 'State', 'Territory']
grade = pd.get_dummies(test['Grade']).iloc[:, 1:]
size = pd.cut(test['Sample_Size'], 1000, labels=False)

test_X = test.iloc[:, [3, 4,7]]
test_X = pd.concat([test_X, race, qcode, sid1, sid2, sid3, year, sex, loc, stype, grade, size], axis=1)

# Fitting regressors
dectree = DecisionTreeRegressor()
forest = RandomForestRegressor(n_estimators=100)

dectree.fit(X, y)
forest.fit(X, y)

gbr = GBR(n_estimators=1000)
gbr.fit(X, y)

# predicting
prediction = dectree.predict(test_X)
prediction = forest.predict(test_X)
prediction = gbr.predict(test_X)

s = pd.read_csv('sample_submission.csv')
sample = pd.DataFrame(columns=['Patient_ID','Greater_Risk_Probability'])
sample['Patient_ID'] = test['Patient_ID']
sample['Greater_Risk_Probability'] = prediction
sample.to_csv('sub-13.csv', index=False)

