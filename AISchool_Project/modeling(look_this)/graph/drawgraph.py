import numpy as np
import plotly.express as px
import pandas as pd

prediction = pd.read_csv('C:/Users/gjaischool/PycharmProjects/ai_school/AISchool_Project/modeling/graph/prediction.csv', header=0, index_col=0)
print(prediction)


prediction2 = prediction.value_counts().reset_index(level=0)
prediction2.columns = ['prediction', 'count']
total = prediction2['count'].sum()
# fig = px.pie(names=prediction2.index, values=prediction2.values)
# fig.update_traces(hole=.3)
# fig.show()

# Creating the pie chart
fig = px.pie(prediction2, names='prediction', values='count', hole=.4, title='리뷰의 긍정-부정 비율')
fig.update_traces(textposition='inside', textinfo='label+percent', textfont_size=20, textfont_color='black')
fig.update_layout(
    annotations=[dict(text=f'리뷰 : {total}개', showarrow=False, font_size=25)]
)
fig.show()