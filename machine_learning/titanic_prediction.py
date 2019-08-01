"""
https://www.kaggle.com/startupsci/titanic-data-science-solutions
for reference
"""


# importing for data wrangling
import pandas as pd
import numpy as np
import random

# importing for data modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

print(train_df.columns.values)
print(train_df.head())
print(train_df.describe(include=['O']))

# checking the survival percentage grouping by class, gender

print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# dropping off unnecessary columns

train_df = train_df.drop(['SibSp', 'Parch', 'Cabin', 'Ticket'], axis=1)
test_df = test_df.drop(['SibSp', 'Parch', 'Cabin', 'Ticket'], axis=1)

# merging test and train data to extract new features

combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                              'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset.Title.replace('Mlle', 'Miss')
    dataset['Title'] = dataset.Title.replace('Ms', 'Miss')
    dataset['Title'] = dataset.Title.replace('Mme', 'Mrs')

print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset.Title.map(title_mapping)
    dataset['Title'] = dataset.Title.fillna(0)

print(train_df.head)

#Dropping off features which are not required

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

for dataset in combine:
    dataset['Sex'] = dataset.Sex.map({'male': 0, 'female': 1}).astype(int)
print(train_df[['Sex', 'Age']])

for dataset in combine:
    guess_df_male = (dataset[(dataset['Sex'] == 0)]['Age'].dropna()).median()
    guess_df_female = (dataset[(dataset['Sex'] == 1)]['Age'].dropna()).median()
    print(guess_df_male, guess_df_female)
    dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == 0), 'Age'] = guess_df_male
    dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == 1), 'Age'] = guess_df_female
print(train_df.head(), test_df.head())

freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset.Embarked.fillna(freq_port)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

for dataset in combine:
    dataset['Fare'] = dataset.Fare.fillna(35)

#space for adding age group correlation


# continuing the code from here

print(train_df.shape, test_df.shape)
random_seed = 2
x_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
x_test = test_df.drop('PassengerId', axis=1).copy()
print(x_train.describe(), '\n', y_train.describe(), '\n', x_test.describe())
# train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.18, shuffle=True, random_state=random_seed)
# print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)

#modelling finally phew!!

decision_tree = DecisionTreeClassifier(max_depth=6)
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train), 2)
print(acc_decision_tree)

random_forest = RandomForestClassifier(max_depth=6)
random_forest.fit(x_train, y_train)
y1_pred = random_forest.predict(x_test)
acc_decision_tree_1 = round(random_forest.score(x_train, y_train), 2)
print(acc_decision_tree_1)

submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y1_pred})
submission.to_csv('submission.csv', index=False)