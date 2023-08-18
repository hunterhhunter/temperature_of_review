import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


class TextDataLoader:
    def __init__(self, train_path, test_path, encoding='cp949'):
        self.train_data = pd.read_csv(train_path, encoding=encoding, header=None)
        self.test_data = pd.read_csv(test_path, encoding=encoding, header=None)
        self.vectorizer = None

    def transform_data(self, max_features, min_df=1, token_pattern=r"\b\w+\b"):
        self.vectorizer = CountVectorizer(max_features=max_features, min_df=min_df, token_pattern=token_pattern)

        X_train = self.vectorizer.fit_transform(self.train_data.iloc[:, 0])
        y_train = self.train_data.iloc[:, 1]

        X_test = self.vectorizer.transform(self.test_data.iloc[:, 0])
        y_test = self.test_data.iloc[:, 1]

        return X_train, y_train, X_test, y_test


class TextClassifier:
    def __init__(self, model_params):
        self.model = MLPClassifier(**model_params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def save_model(self, path):
        with open(path, 'wb') as f:
            joblib.dump(self.model, f)


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        results = {
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc_score': roc_auc_score(y_true, y_pred)
        }
        return results


if __name__ == "__main__":
    train_path = 'path_to_train_data'
    test_path = 'path_to_test_data'

    data_loader = TextDataLoader(train_path, test_path)
    X_train, y_train, X_test, y_test = data_loader.transform_data(max_features=18203)

    classifier = TextClassifier(
        model_params={'max_iter': 1000, 'alpha': 1, 'learning_rate': 'adaptive', 'hidden_layer_sizes': (100, 100, 100),
                      'random_state': 42})
    classifier.train(X_train, y_train)

    y_pred = classifier.predict(X_test)
    results = ModelEvaluator.evaluate(y_test, y_pred)
    for key, value in results.items():
        print(f"{key}: {value}")

    classifier.save_model('path_to_save_model')

# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
#
#
# train_data = pd.read_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/morphed_train.csv', encoding='cp949', header=None)
# test_data = pd.read_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/morphed_test.csv', encoding='cp949', header=None)
#
# X_train, y_train = train_data.iloc[:, 0], train_data.iloc[:, 1]
# X_test, y_test = test_data.iloc[:, 0], test_data.iloc[:, 1]
#
# from sklearn.feature_extraction.text import CountVectorizer
# vect = CountVectorizer(max_features=18203,min_df=1, token_pattern=r"\b\w+\b").fit(X_train)
# X_train = vect.transform(X_train)
# X_test = vect.transform(X_test)
# print("X_train:/n", repr(X_train))
# print("X_train:/n", repr(X_train.toarray()))
#
# print(train_data.shape)
# print(test_data.shape)
# print(X_train.shape)
# print(X_test.shape)
#
# model = MLPClassifier(max_iter=1000, alpha=1,learning_rate='adaptive', hidden_layer_sizes=(100, 100, 100), random_state=42)
# model.fit(X_train, y_train)
#
#
# print("prediction:", model.predict(X_test))
# print('훈련 데이터 점수:', model.score(X_train, y_train))
# print("테스트데이터 점수:", model.score(X_test, y_test))
# print("테스트데이터 점수(반올림):", round(model.score(X_test, y_test), 3))
#
# print()
#
# results = confusion_matrix(y_true = y_test, y_pred = model.predict(X_test))
# print(results)
#
# print()
#
# prediction = model.predict(X_test)
#
# precision = precision_score(y_test, prediction)
# print('precision:', precision)
# recall = recall_score(y_test, prediction)
# print('recall:', recall)
# f1 = f1_score(y_test, prediction)
# print('f1 score:', f1)
#
# print()
#
# roc_auc = roc_auc_score(y_test, prediction)
# print('ROC AUC score:', roc_auc)
#
# print()
#
# with open('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/models/vectorizer', 'wb') as f:
#     joblib.dump(vect, f)
#
# with open('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/models/mlp_model', 'wb') as f:
#     joblib.dump(model, f)