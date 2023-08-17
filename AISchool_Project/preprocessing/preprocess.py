import pandas as pd

data = pd.read_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/preprocessing/reviwe_final_raw.csv', header=0, encoding='cp949')

print(data.head())
print()
print(data.shape)