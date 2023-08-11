import pandas as pd
from collections import Counter
from collections import defaultdict

# 리뷰 데이터를 불러옵니다.
morphed_data = pd.read_csv("C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/morphed_data_with_prodname.csv", encoding='cp949')
# 결측값이 있는 경우 빈 문자열로 대체합니다.
morphed_data['0'] = morphed_data['0'].fillna('')

# 긍정 리뷰에서 사용될 높은 빈도의 키워드를 불러옵니다.
with open("C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/pos_highfreq_keyword.txt", 'r', encoding='utf-8') as f:
    high_freq_keywords = [line.strip() for line in f.readlines()]

# 부정 리뷰에서 사용될 높은 빈도의 키워드를 불러옵니다.
with open("C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/neg_highfreq_keyword.txt", 'r', encoding='utf-8') as f:
    neg_high_freq_keywords = [line.strip() for line in f.readlines()]

# 상품별로 키워드와 리뷰 개수를 저장할 딕셔너리를 초기화합니다.
product_keyword_counts_detailed_both_updated = defaultdict(lambda: defaultdict(int))

# 상품명으로 그룹화하여 각 상품에 대해 분석을 진행합니다.
for product, group in morphed_data.groupby('상품명'):
    # 전체 리뷰 개수, 긍정 리뷰 개수, 부정 리뷰 개수를 계산합니다.
    total_reviews = len(group)
    positive_group = group[group['구매자 평점'] == 1]
    negative_group = group[group['구매자 평점'] == 0]
    total_positive_reviews = len(positive_group)
    total_negative_reviews = len(negative_group)
    # 결과를 딕셔너리에 저장합니다.
    product_keyword_counts_detailed_both_updated[product]['total_reviews'] = total_reviews
    product_keyword_counts_detailed_both_updated[product]['total_positive_reviews'] = total_positive_reviews
    product_keyword_counts_detailed_both_updated[product]['total_negative_reviews'] = total_negative_reviews
    
    # 긍정 리뷰에서 긍정 키워드의 빈도를 계산합니다.
    for review in positive_group['0']:
        for keyword in high_freq_keywords:
            if keyword in review:
                product_keyword_counts_detailed_both_updated[product][f'pos_{keyword}'] += 1

    # 부정 리뷰에서 부정 키워드의 빈도를 계산합니다.
    for review in negative_group['0']:
        for keyword in neg_high_freq_keywords:
            if keyword in review:
                product_keyword_counts_detailed_both_updated[product][f'neg_{keyword}'] += 1

