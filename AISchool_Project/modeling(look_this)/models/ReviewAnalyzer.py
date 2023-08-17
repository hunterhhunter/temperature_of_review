import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import rhinoMorph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# RhinoMorph 형태소 분석을 적용하는 변환기 클래스를 정의합니다.
class RhinoMorphTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rn = rhinoMorph.startRhino()  # RhinoMorph 형태소 분석기를 시작합니다.

    def fit(self, X, y=None):
        return self  # 이 변환기는 학습이 필요 없으므로 그대로 self를 반환합니다.

    def transform(self, X):
        # X의 각 원소에 RhinoMorph 형태소 분석을 적용하고, 토큰을 공백으로 연결한 문자열을 반환합니다.
        transformed = [' '.join(rhinoMorph.onlyMorph_list(self.rn, text, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'], eomi=True)) for text in X]
        return transformed

# 리뷰 분석을 수행하는 클래스를 정의합니다.
class ReviewAnalyzer:
    def __init__(self, data_path, model_path, scaler_path):
        self.data_path = data_path  # 데이터 파일 경로
        self.model_path = model_path  # 모델 파일 경로
        self.scaler_path = scaler_path  # 스케일러 파일 경로
        self.rhinomorph = RhinoMorphTransformer()  # RhinoMorph 변환기 인스턴스
        self.loaded_model = joblib.load(model_path)  # 모델 로드
        self.loaded_scaler = joblib.load(scaler_path)  # 스케일러 로드
        # 파이프라인 구성
        self.pipeline = Pipeline(steps=[
            ('rhinomorph', self.rhinomorph),
            ('scaler', self.loaded_scaler),
            ('model', self.loaded_model)
        ])

    def load_data(self):
        # 데이터를 로드합니다.
        self.raw_data = pd.read_csv(self.data_path, encoding='cp949', header=0, index_col=0)
        self.text_data = self.raw_data.iloc[:, 0].astype(str)  # 리뷰 텍스트
        self.target = self.raw_data.iloc[:, 1]  # 리뷰의 긍정/부정 라벨

    def transform_data(self):
        # 리뷰 텍스트를 형태소 분석을 통해 변환합니다.
        self.transformed_data = self.rhinomorph.transform(self.text_data)

    def predict(self):
        # 변환된 리뷰 텍스트를 이용해 긍정/부정을 예측합니다.
        self.prediction = self.pipeline.predict(self.text_data)
        self.prediction_label = np.where(self.prediction > 0, '긍정', '부정')  # 0보다 크면 '긍정', 아니면 '부정'

    def visualize(self):
        # 예측 결과를 시각화합니다.
        prediction_df = pd.Series(self.prediction_label)
        prediction_counts = prediction_df.value_counts().reset_index(level=0)
        prediction_counts.columns = ['prediction', 'count']
        total = prediction_counts['count'].sum()

        fig = px.pie(prediction_counts, names='prediction', values='count', hole=.4, title='리뷰의 긍정-부정 비율')
        fig.update_traces(textposition='inside', textinfo='label+percent', textfont_size=15, textfont_color='black')
        fig.update_layout(annotations=[dict(text=f'전체 리뷰 개수 : {total}', showarrow=False, font_size=15)])
        fig.show()

    def analyze(self):
        # 리뷰 분석 전체 과정을 수행합니다.
        self.load_data()
        self.transform_data()
        self.predict()
        self.visualize()

# 사용 예시
# 객체 생성
analyzer = ReviewAnalyzer('데이터 경로', '모델 불러오는 경로', '전처리 모델 경로')
analyzer.analyze()
