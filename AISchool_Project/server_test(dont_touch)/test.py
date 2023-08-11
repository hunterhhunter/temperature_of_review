from test_server.ReviewAnalyzer_final import ProductReviewAnalyzer

data_path = 'C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/test_server/data/serv_test.csv'
model_path = 'C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/test_server/modeling/models/mlp_model'
scaler_path = 'C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/test_server/modeling/models/vectorizer'
pos_keywords = 'C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/test_server/modeling/data/pos_highfreq_keyword.txt'
neg_keywords = 'C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/test_server/modeling/data/neg_highfreq_keyword.txt'

analyzer = ProductReviewAnalyzer(data_path, model_path, scaler_path, pos_keywords, neg_keywords)
analyzer.analyze()