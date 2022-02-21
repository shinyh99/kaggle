# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print(train.isnull().sum() / len(train))
# print(test.isnull().sum() / len(test))

seaborn.set()

# seaborn.countplot(data=train, x="Survived", hue="Sex")
# seaborn.countplot(data=train, x="Sex", hue="Survived")


def draw_bar_chart(feature):
    survived = train[train["Survived"] == 1][feature].value_counts()
    dead = train[train["Survived"] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ["Survived", "Dead"]
    df.plot(kind="bar", stacked=True, figsize=(10, 5))


# draw_bar_chart("Sex")
# draw_bar_chart("Embarked")
# draw_bar_chart("Pclass")

# train.groupby(["Sex", "Embarked", "Pclass"])["Survived"].mean()

train["Age"].describe()


def draw_hist_chart(feature):
    plt.figure(figsize=(10, 5))
    seaborn.histplot(
        train[train["Survived"] == 1][feature].dropna(), color="b", kde=True
    )
    seaborn.histplot(
        train[train["Survived"] == 0][feature].dropna(), color="r", kde=True
    )
    plt.legend(["Survived", "Died"])


# draw_hist_chart("Age")

# draw_hist_chart("Fare")
# plt.xlim([0, 50])

# plt.figure(figsize=(8, 6))
# seaborn.scatterplot(data=train, x="Fare", y="Age")


# * 결측 데이터 채우기
feature = pd.concat([train, test])
feature = feature.reset_index(drop=True)
# print(feature.isna().sum() / len(feature))

# Fill embarked
# print(feature["Embarked"].value_counts())
# print(feature["Embarked"].value_counts(normalize=True))
# seaborn.countplot(feature["Embarked"])
feature["Embarked"] = feature["Embarked"].fillna("S")

# Fill Fare
# seaborn.histplot(feature["Fare"], kde=True)
feature["Fare"] = feature["Fare"].fillna(feature["Fare"].median())

# print(feature.isna().sum())

# Core Fare that is zero
# len(feature[feature["Fare"] == 0])
feature.groupby(["Pclass"]).median()
feature.loc[(feature["Fare"] == 0) & (feature["Pclass"] == 1), "Fare"] = 60.0
feature.loc[(feature["Fare"] == 0) & (feature["Pclass"] == 2), "Fare"] = 15.0458
feature.loc[(feature["Fare"] == 0) & (feature["Pclass"] == 3), "Fare"] = 8.0500
# feature["Fare"].describe()

# Fill Age
feature.groupby(["Sex", "Pclass"]).mean()
feature.loc[
    (feature["Age"].isna())
    & (feature["Sex"] == "female")
    & (feature["Pclass"] == 1),
    "Age",
] = 37
feature.loc[
    (feature["Age"].isna())
    & (feature["Sex"] == "female")
    & (feature["Pclass"] == 2),
    "Age",
] = 27.5
feature.loc[
    (feature["Age"].isna())
    & (feature["Sex"] == "female")
    & (feature["Pclass"] == 3),
    "Age",
] = 22.2
feature.loc[
    (feature["Age"].isna())
    & (feature["Sex"] == "male")
    & (feature["Pclass"] == 1),
    "Age",
] = 41
feature.loc[
    (feature["Age"].isna())
    & (feature["Sex"] == "male")
    & (feature["Pclass"] == 2),
    "Age",
] = 30.8
feature.loc[
    (feature["Age"].isna())
    & (feature["Sex"] == "male")
    & (feature["Pclass"] == 3),
    "Age",
] = 26

# feature.isna().sum()

# Drop some features
del feature["Cabin"], feature["Ticket"]

# Change Sex string to bool
feature["Sex_1"] = feature["Sex"].apply(lambda x: 1 if x == "male" else 0)

# Sibling and Parnet & Children
feature["family"] = feature["SibSp"] + feature["Parch"]
# seaborn.countplot(data=feature, x="family", hue="Survived")

# map to alone, small, big family
family_map = {
    0: "alone",
    1: "small_family",
    2: "small_familiy",
    3: "small_familiy",
    4: "big_familiy",
    5: "big_familiy",
    6: "big_familiy",
    7: "big_familiy",
    10: "big_familiy",
}
feature["family_1"] = feature["family"].map(family_map)

# Retrieve title from name
feature["title"] = feature["Name"].apply(
    lambda x: x.split(",")[1].split(".")[0].strip()
)
# feature["title"].unique()
# feature["title"].value_counts()
title_map = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Rev": "Others",
    "Dr": "Others",
    "Col": "Others",
    "Mlle": "Others",
    "Major": "Others",
    "Ms": "Others",
    "Lady": "Others",
    "Sir": "Others",
    "Mme": "Others",
    "Don": "Others",
    "Capt": "Others",
    "the Countess": "Others",
    "Jonkheer": "Others",
    "Dona": "Others",
}
feature["title_1"] = feature["title"].map(title_map)

# Drop converted feature
del (
    feature["Parch"],
    feature["SibSp"],
    feature["Name"],
    feature["Sex"],
    feature["title"],
    feature["family"],
)

# Normalize Fare
# plt.figure(figsize=(8, 6))
# seaborn.displot(feature["Fare"])
feature["Fare_log"] = np.log(feature["Fare"])
# seaborn.displot(feature["Fare_log"])

# Normalize Age
# seaborn.displot(feature["Age"])
feature["Age_n"] = (
    (feature["Age"] - feature["Age"].min())
    / (feature["Age"].max() - feature["Age"].min())
    * 3
)
# seaborn.displot(feature["Age_n"])

# drop
del feature["Fare"], feature["Age"]

# * One-Hot incoding
feature2 = pd.get_dummies(
    feature,
    columns=["Embarked", "family_1", "title_1", "Pclass"],
    drop_first=True,
)


# * KNN ML
# Train and Test Set
train_f = feature2.dropna()
test_f = feature2[feature2["Survived"].isna()]
del test_f["Survived"]

target = train_f["Survived"]
del train_f["Survived"], train_f["PassengerId"]

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
model = KNeighborsClassifier(n_neighbors=10)
scoring = "accuracy"
score = cross_val_score(
    model, train_f, target, cv=k_fold, n_jobs=1, scoring=scoring
)
# print(score)
# round(np.mean(score) * 100, 2)
model = KNeighborsClassifier(n_neighbors=10)
model.fit(train_f, target)

test_data = test_f.drop("PassengerId", axis=1).copy()
prediction = model.predict(test_data)
submission = pd.DataFrame(
    {"PassengerId": test_f["PassengerId"], "Survived": prediction.astype(int)}
)

submission.to_csv("submission_KNN.csv", index=False)


# * SVM ML
model = SVC()
scoring = "accuracy"
score = cross_val_score(
    model, train_f, target, cv=k_fold, n_jobs=1, scoring=scoring
)

model = SVC()
model.fit(train_f, target)
test_data = test_f.drop("PassengerId", axis=1).copy()
prediction = model.predict(test_data)

submission = pd.DataFrame(
    {"PassengerId": test_f["PassengerId"], "Survived": prediction.astype(int)}
)
submission.to_csv("submission_SVC.csv", index=False)

# * Decision Tree
model = DecisionTreeClassifier()
scoring = "accuracy"
score = cross_val_score(
    model, train_f, target, cv=k_fold, n_jobs=1, scoring=scoring
)

model = DecisionTreeClassifier()
model.fit(train_f, target)
test_data = test_f.drop("PassengerId", axis=1).copy()
prediction = model.predict(test_data)

submission = pd.DataFrame(
    {"PassengerId": test_f["PassengerId"], "Survived": prediction.astype(int)}
)
submission.to_csv("submission_DT.csv", index=False)

# * SVC Ensemble
model = BaggingClassifier(
    base_estimator=SVC(), n_estimators=100, random_state=0
)
model.fit(train_f, target)
test_data = test_f.drop("PassengerId", axis=1).copy()
prediction = model.predict(test_data)
submission = pd.DataFrame(
    {"PassengerId": test_f["PassengerId"], "Survived": prediction.astype(int)}
)
submission.to_csv("submission_bag_SVC.csv", index=False)

# * KNN Ensemble
model = BaggingClassifier(
    base_estimator=KNeighborsClassifier(), n_estimators=100, random_state=0
)
model.fit(train_f, target)
test_data = test_f.drop("PassengerId", axis=1).copy()
prediction = model.predict(test_data)
submission = pd.DataFrame(
    {"PassengerId": test_f["PassengerId"], "Survived": prediction.astype(int)}
)
submission.to_csv("submission_bag_KNN.csv", index=False)
