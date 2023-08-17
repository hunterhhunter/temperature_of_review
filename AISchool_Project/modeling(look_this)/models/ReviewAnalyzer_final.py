
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import rhinoMorph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from collections import defaultdict
from typing import List, Dict, Union

# RhinoMorph 형태소 분석을 적용하는 변환기 클래스를 정의합니다.
class RhinoMorphTransformer(BaseEstimator, TransformerMixin):
    """형태소 분석 클래스"""
    def __init__(self):
        self.rn = rhinoMorph.startRhino()

    def fit(self, X: List[str], y=None) -> 'RhinoMorphTransformer':
        return self

    def transform(self, X: pd.Series) -> List[str]:
        transformed = [' '.join(rhinoMorph.onlyMorph_list(self.rn, text, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'], eomi=True)) for text in X]
        return transformed

# 리뷰 분석을 수행하는 기본 클래스를 정의합니다.
class BaseReviewAnalyzer:
    """기본 리뷰 분석 클래스"""

    def __init__(self, data_path: str, model_path: str, scaler_path: str):
        self.data_path = data_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.rhinomorph = RhinoMorphTransformer()
        self.loaded_model = joblib.load(model_path)
        self.loaded_scaler = joblib.load(scaler_path)
        self.pipeline = Pipeline(steps=[
            ('scaler', self.loaded_scaler),
            ('model', self.loaded_model)
        ])

    def load_data(self) -> None:
        """데이터 로드 메서드"""
        self.raw_data = pd.read_csv(self.data_path, encoding='cp949', header=0, index_col=0)
        self.text_data = self.raw_data.iloc[:, 1].astype(str)

    def transform_data(self) -> None:
        """데이터 변환 메서드"""
        self.transformed_texts = self.rhinomorph.transform(self.text_data)
        self.transformed_data = pd.DataFrame({
            '상품명': self.raw_data['상품명'], # 상품명 칼럼이 있다고 가정합니다.
            '리뷰': self.transformed_texts
        })

    def save_transform_data(self, data: pd.DataFrame) -> None:
        """변환된 데이터 저장 메서드"""
        data.to_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/transform_data.csv', encoding='cp949')

    def predict(self) -> None:
        """예측 수행 메서드"""
        self.prediction = self.pipeline.predict(self.transformed_texts)
        print(self.prediction.sum())
        self.prediction_label = np.where(self.prediction > 0, '긍정', '부정')
        self.transformed_data['구매자 평점'] = self.prediction

    def analyze(self) -> None:
        """전체 분석 프로세스 수행 메서드"""
        self.load_data()
        self.transform_data()
        self.predict()

# 상품별 리뷰 분석 클래스를 정의합니다.
class ProductReviewAnalyzer(BaseReviewAnalyzer):
    """상품별 리뷰 분석 클래스"""
    def __init__(self, data_path: str, model_path: str, scaler_path: str, pos_highfreq_keywords_path: str, neg_highfreq_keywords_path: str):
        super().__init__(data_path, model_path, scaler_path)

        # 긍정 리뷰에서 사용될 높은 빈도의 키워드를 불러옵니다.
        with open(pos_highfreq_keywords_path, 'r', encoding='utf-8') as f:
            self.pos_highfreq_keywords = [line.strip() for line in f.readlines()]

        # 부정 리뷰에서 사용될 높은 빈도의 키워드를 불러옵니다.
        with open(neg_highfreq_keywords_path, 'r', encoding='utf-8') as f:
            self.neg_highfreq_keywords = [line.strip() for line in f.readlines()]

    def analyze(self) -> None:
        """전체 분석 프로세스 수행 메서드"""
        super().analyze()
        product_review_analyzer_helper = ProductReviewAnalyzerHelper(self.transformed_data, self.pos_highfreq_keywords, self.neg_highfreq_keywords)
        product_keyword_counts_detailed = product_review_analyzer_helper.analyze_reviews()
        self.visualize(product_keyword_counts_detailed)

    def visualize(self, product_keyword_counts_detailed: Dict[str, Dict[str, Union[str, int]]]) -> None:
        """시각화 메서드: 각 상품별 리뷰의 긍정/부정 비율을 시각화합니다."""
    # 예측 결과를 시각화합니다.
        # prediction_df = pd.Series(self.prediction_label)
        # prediction_counts = prediction_df.value_counts().reset_index(level=0)
        # prediction_counts.columns = ['prediction', 'count']
        # total = prediction_counts['count'].sum()

        # fig = px.pie(prediction_counts, names='prediction', values='count', hole=.4, title='리뷰의 긍정-부정 비율')
        # fig.update_traces(textposition='inside', textinfo='label+percent', textfont_size=15, textfont_color='black')
        # fig.update_layout(annotations=[dict(text=f'전체 리뷰 개수 : {total}', showarrow=False, font_size=15)])
        # fig.show()


        sorted_products = sorted(product_keyword_counts_detailed.items(), key=lambda x: x[1]['total_reviews'], reverse=True)
        
        #상품별 상세 정보를 표시합니다.
        for product, details in sorted_products[:5]:
            product_counts = pd.DataFrame({
            'prediction': ['긍정', '부정'],
            'count': [details['total_positive_reviews'], details['total_negative_reviews']]
            })

            fig_product = px.pie(product_counts, names='prediction', values='count', hole=.4, title=f'{product} 리뷰의 긍정-부정 비율')
            fig_product.update_traces(textposition='inside', textinfo='label+percent', textfont_size=20, textfont_color='black')
            fig_product.update_layout(annotations=[dict(text=f"전체 리뷰 개수 : {details['total_reviews']} 개", showarrow=False, font_size=17)])
            fig_product.show()

            print(f"상품명: {product}")
            print(f"총 리뷰 수: {details['total_reviews']}")
            print(f"긍정 리뷰 수: {details['total_positive_reviews']}")
            print(f"부정 리뷰 수: {details['total_negative_reviews']}")
            print("긍정 키워드:")
            for key, value in details.items():
                if key.startswith('pos_'):
                    print(f"  {key[4:]}: {value}")
            print("부정 키워드:")
            for key, value in details.items():
                if key.startswith('neg_'):
                    print(f"  {key[4:]}: {value}")
            print("-" * 50)

class ProductReviewAnalyzerHelper:
    """상품 리뷰 빈도분석 클래스"""
    def __init__(self, transformed_data: pd.DataFrame, pos_highfreq_keywords: List[str], neg_highfreq_keywords: List[str]):
        self.morphed_data = transformed_data  # 형태소 분석 데이터와 상품명이 함께 있는 DataFrame
        self.pos_highfreq_keywords = pos_highfreq_keywords
        self.neg_highfreq_keywords = neg_highfreq_keywords
        self.product_keyword_counts_detailed = defaultdict(lambda: defaultdict(int))

    def analyze_reviews(self) -> Dict[str, Dict[str, Union[str, int]]]:
        """각 상품별 리뷰 분석 메서드"""
        for product, group in self.morphed_data.groupby('상품명'):
            total_reviews = len(group)
            positive_group = group[group['구매자 평점'] == 1]
            negative_group = group[group['구매자 평점'] == 0]
            total_positive_reviews = len(positive_group)
            total_negative_reviews = len(negative_group)
            self.product_keyword_counts_detailed[product]['total_reviews'] = total_reviews
            self.product_keyword_counts_detailed[product]['total_positive_reviews'] = total_positive_reviews
            self.product_keyword_counts_detailed[product]['total_negative_reviews'] = total_negative_reviews

            pos_keyword_counts = defaultdict(int)
            neg_keyword_counts = defaultdict(int)

            for review in positive_group['리뷰']:
                for keyword in self.pos_highfreq_keywords:
                    if keyword in review:
                        pos_keyword_counts[keyword] += 1

            for review in negative_group['리뷰']:
                for keyword in self.neg_highfreq_keywords:
                    if keyword in review:
                        neg_keyword_counts[keyword] += 1

            # 상위 5개 긍정 키워드 선택
            top_5_pos_keywords = sorted(pos_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for keyword, count in top_5_pos_keywords:
                self.product_keyword_counts_detailed[product][f'pos_{keyword}'] = count

            # 상위 5개 부정 키워드 선택
            top_5_neg_keywords = sorted(neg_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for keyword, count in top_5_neg_keywords:
                self.product_keyword_counts_detailed[product][f'neg_{keyword}'] = count

        return self.product_keyword_counts_detailed