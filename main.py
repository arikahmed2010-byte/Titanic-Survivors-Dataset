# Import Cell

import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection, metrics

# Appropriate Training Data Conversion Cell

train = pd.read_csv("train.csv").dropna()

# Sex Conversion to boolean numbers (male -> 0, female -> 1)
train["Sex"] = train["Sex"].map({
    "male": 0,
    "female": 1
})

# Embarked Conversion to boolean numbers (S -> 0, Q -> 1, C -> 2)
train["Embarked"] = train["Embarked"].map({
    "S": 0,
    "Q": 1,
    "C": 2
})

X = train[
    [
      "Pclass",
      "Sex",
      "Age",
      "SibSp",
      "Parch",
      "Fare",
      "Embarked"
    ]
]

y = train["Survived"]

# Model Initialization & Training

model = linear_model.LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy Testing

accuracy_score = metrics.accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy Score: {accuracy_score}")

test = pd.read_csv("test.csv").dropna()

test["Sex"] = test["Sex"].map({
    "male": 0,
    "female": 1
})

test["Embarked"] = test["Embarked"].map({
    "S": 0,
    "Q": 1,
    "C": 2
})

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

test["Survived"] = model.predict(test[features])

test.to_csv("predictions.csv", index=False)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
