import rhinoMorph
import pandas as pd

data = pd.read_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/data_with_prodname_phone.csv', encoding='cp949', header=0, index_col=0)

data = data.sort_values('상품명')
data.reset_index(drop=True)

prodname = data.iloc[:, 0]
rating = data.iloc[:, 2]
text = data.iloc[:, 1]
text = text.astype(str)
text = text.values
rn = rhinoMorph.startRhino()

morphed_text = [' '.join(rhinoMorph.onlyMorph_list(rn, text_i, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'], eomi=True)) for text_i in text]
morphed_text = pd.Series(morphed_text)

morphed_data_prodname = pd.concat([prodname, morphed_text, rating], axis=1)
print(morphed_data_prodname)
morphed_data_prodname.to_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/data/morphed_data_with_prodname.csv', encoding='cp949')