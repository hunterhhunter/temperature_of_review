from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import pandas as pd
import rhinoMorph
import plotly.express as px

loaded_scaler = joblib.load('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/models/vectorizer')
loaded_model = joblib.load('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/models/mlp_model')

raw_data = pd.read_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/data_no_prodname_phone.csv', encoding='cp949', header=0, index_col=0)

class RhinoMorphTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rn = rhinoMorph.startRhino()  # RhinoMorph 형태소 분석기 시작

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply RhinoMorph to each element in X and join the tokens with a space
        transformed = [' '.join(rhinoMorph.onlyMorph_list(self.rn, text, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'], eomi=True)) for text in X]
        self.transformed = transformed  # Store the transformed data
        return transformed
    
rhinomorph = RhinoMorphTransformer()
pipeline = Pipeline(steps=[
    ('rhinomorph', rhinomorph),
    ('scaler', loaded_scaler),
    ('model', loaded_model)
])

text_data = raw_data.iloc[:, 0]
target = raw_data.iloc[:, 1]
text_data = text_data.astype(str)

prediction = pipeline.predict(text_data)
prediction3 = np.where(prediction > 0, '긍정', '부정')
print(len(prediction))
print(prediction.sum())
matches = prediction == target
acc = matches.mean()
print('ACC :', acc)

print(type(prediction))

prediction2 = pd.Series(prediction)
text_data = pd.Series(rhinomorph.transformed)
morphed_data_with_predict = pd.concat([text_data, prediction2], axis=1)

morphed_data_with_predict.to_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/text_data_with_predict.csv')

prediction = pd.read_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/graph/prediction.csv', header=0, index_col=0)
print(prediction)


# prediction2 = prediction.value_counts().reset_index(level=0)
# prediction2.columns = ['prediction', 'count']
# total = prediction2['count'].sum()
# # fig = px.pie(names=prediction2.index, values=prediction2.values)
# # fig.update_traces(hole=.3)
# # fig.show()

# # Creating the pie chart
# fig = px.pie(prediction2, names='prediction', values='count', hole=.4, title='리뷰의 긍정-부정 비율')
# fig.update_traces(textposition='inside', textinfo='label+percent', textfont_size=15, textfont_color='black')
# fig.update_layout(
#     annotations=[dict(text=f'전체 리뷰 개수 : {total}', showarrow=False, font_size=15)]
# )
# fig.show()