import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

"""
Group 71 :
Rahul(2019191)
Raghav Sharma(2019189)
Abhish Panwar(2019135)

"""

"""
Explaination of Algorithm :

    Assumptions : As we go through the given train data found that the last sequence is valid peptide
                  because it contains character('/',u, etc) which do not defines any amino acid, therefore
                  we are removing those sequences Only from train data, as we assume no such sequence will be
                  exist in test data.

    Feature Generation : We used the method of composition to generate feature by two ways :

            Method1 : Generated the feature using frequency of the single character in a sequence divided by the length of the sequence.
            Method2 : Generated feature by taking two adjacent amino acid frequency in a sequence divided by the length of the sequence.
                      Example :
                        AFAFFAGHI

                        we have freq(AF) = 2
                        therefore, we will have 2/(9) as the position of AF

                      and we have in total 20*20 = 400(adjancent) + 20(single) features for a sequence


    Model training : After trying models like, SVM, KNN, MLP,CatergoricalNB, Random Forest and many more we found that
                     Random Forest is giving us the most desired results. with max_depth = 60 and n_estimatores = 1000+

"""

# Taking Input in specified format, and the files has to be in same directory
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Remove the sequence which are not valid for an peptide sequence(having amino acid other than those specified 20)
train_two = []
cur = 0
for i in range(train_data.shape[0]):
    z = len(train_data.iloc[i][0])
    x = 0
    for j in range(z):
        y = (ord(train_data.iloc[i][0][j]) - 65)
        if ((y < 0 or y > 25) == True):
            x = 1
            break
    if (x == 0):
        train_two.append(train_data.iloc[i])
        cur += 1

train_two = pd.DataFrame(train_two)
train_data = train_two

# Generating the features with only single character.

train_feature20 = []
test_feature20 = []

train_label = train_data[' Label']

for i in range(train_data.shape[0]):
    x = []
    for j in range(26):
        x.append(0)
    z = len(train_data.iloc[i][0])
    for j in range(z):
        y = (ord(train_data.iloc[i][0][j]) - 65)
        x[y] = x[y] + 1

    for j in range(26):
        x[j] = x[j] / z

    train_feature20.append(x)

# Generating features are using single character for test data

for i in range(test_data.shape[0]):
    x = []
    for j in range(26):
        x.append(0)
    z = len(test_data.iloc[i][1])
    for j in range(z):
        y = (ord(test_data.iloc[i][1][j]) - 65)
        x[y] = x[y] + 1
    for j in range(26):
        x[j] = x[j] / z

    test_feature20.append(x)

test_feature20 = pd.DataFrame(test_feature20)
train_feature20 = pd.DataFrame(train_feature20)
test_feature2020 = []
test_feature2020 = pd.DataFrame(test_feature2020)
train_feature2020 = []
train_feature2020 = pd.DataFrame(train_feature2020)

test_feature2020 = test_feature20[0]
train_feature2020 = train_feature20[0]

# Remove columns which are invalid sequence such as B,U,O, etc.

for i in range(1, 26):
    if (i != 1 and i != 23 and i != 25 and i != (ord('J') - 65) and i != (ord('O') - 65) and i != (ord('U') - 65)):
        test_feature2020 = pd.concat([test_feature2020, test_feature20[i]], axis=1)
        train_feature2020 = pd.concat([train_feature2020, train_feature20[i]], axis=1)

# Find the invalid sequeunce for two adjacent characters.

arr = []
notar = ['B', 'X', 'Z', 'O', 'J', 'U']

notadd = []

for i in range(26):
    for j in range(6):
        notadd.append(i * 26 + (ord(notar[j]) - 65))
        notadd.append((ord(notar[j]) - 65) * 26 + i)
train_feature = []
test_feature = []

# Finding the adjancent characters in the seuquence and appending the value at the position
for i in range(train_data.shape[0]):
    x = []
    for j in range(26 * 26):
        x.append(0)
    z = len(train_data.iloc[i][0])
    for j in range(z - 1):
        y = (ord(train_data.iloc[i][0][j]) - 65)
        p = (ord(train_data.iloc[i][0][j + 1]) - 65)
        pos = ((y) * 26) + p

        x[pos] = x[pos] + 1

    for j in range(26 * 26):
        x[j] = x[j] / z

    train_feature.append(x)

for i in range(test_data.shape[0]):
    x = []
    for j in range(26 * 26):
        x.append(0)
    z = len(test_data.iloc[i][1])
    for j in range(z - 1):
        y = (ord(test_data.iloc[i][1][j]) - 65)
        p = (ord(test_data.iloc[i][1][j + 1]) - 65)
        pos = ((y) * 26) + p

        x[pos] = x[pos] + 1
    for j in range(26 * 26):
        x[j] = x[j] / z

    test_feature.append(x)

test_feature = pd.DataFrame(test_feature)
train_feature = pd.DataFrame(train_feature)
test_feature1 = []
test_feature1 = pd.DataFrame(test_feature1)
train_feature1 = []
train_feature1 = pd.DataFrame(train_feature1)
import pandas as pd

test_feature1 = test_feature[0]
train_feature1 = train_feature[0]

# Removing invalid columns from the 2 character features
for i in range(1, 26 * 26):
    if ((i in notadd) == False):
        test_feature1 = pd.concat([test_feature1, test_feature[i]], axis=1)
        train_feature1 = pd.concat([train_feature1, train_feature[i]], axis=1)
test_feature1 = pd.concat([test_feature1, test_feature2020], axis=1)
train_feature1 = pd.concat([train_feature1, train_feature2020], axis=1)

i = 0;
y_final = []

while (i != 1):
    gnb = RandomForestClassifier(max_depth=60, random_state=83, max_features='sqrt', n_estimators=2000)
    gnb.fit(train_feature1, train_label)
    y_pred = gnb.predict_proba(test_feature1)
    y_final.append(y_pred)
    i += 1
ans=[]
for i in range(test_data.shape[0]):
  ans.append(y_final[0][i][1])
#Creating submission file here

Squence_ID = []

for i in range(10001, 10001 + len(ans)):
    Squence_ID.append(i)

Squence_ID = pd.DataFrame(Squence_ID)
FinalResult = pd.DataFrame(ans)

final_data = pd.concat([Squence_ID, FinalResult], axis=1)
final_data.columns = ['ID', 'label']
final_data.to_csv('Submission.csv', index=0)