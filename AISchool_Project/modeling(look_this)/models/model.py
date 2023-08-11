import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


train_data = pd.read_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/morphed_train.csv', encoding='cp949', header=None)
test_data = pd.read_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/morphed_test.csv', encoding='cp949', header=None)

X_train, y_train = train_data.iloc[:, 0], train_data.iloc[:, 1]
X_test, y_test = test_data.iloc[:, 0], test_data.iloc[:, 1]

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features=18203,min_df=1, token_pattern=r"\b\w+\b").fit(X_train)
X_train = vect.transform(X_train)
X_test = vect.transform(X_test)
print("X_train:/n", repr(X_train))
print("X_train:/n", repr(X_train.toarray()))

print(train_data.shape)
print(test_data.shape)
print(X_train.shape)
print(X_test.shape)

model = MLPClassifier(max_iter=1000, alpha=1,learning_rate='adaptive', hidden_layer_sizes=(100, 100, 100), random_state=42)
model.fit(X_train, y_train)


print("prediction:", model.predict(X_test))
print('훈련 데이터 점수:', model.score(X_train, y_train))
print("테스트데이터 점수:", model.score(X_test, y_test))
print("테스트데이터 점수(반올림):", round(model.score(X_test, y_test), 3))

print()

results = confusion_matrix(y_true = y_test, y_pred = model.predict(X_test))
print(results)

print()

prediction = model.predict(X_test)

precision = precision_score(y_test, prediction)
print('precision:', precision)
recall = recall_score(y_test, prediction)
print('recall:', recall)
f1 = f1_score(y_test, prediction)
print('f1 score:', f1)

print()

roc_auc = roc_auc_score(y_test, prediction)
print('ROC AUC score:', roc_auc)

print()

with open('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/models/vectorizer', 'wb') as f:
    joblib.dump(vect, f)

with open('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/models/mlp_model', 'wb') as f:
    joblib.dump(model, f)