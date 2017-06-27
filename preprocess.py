import pandas as pd

class_age_to_num = {'YOUNG':0, 'MIDDLE':1, 'OLD':2}

dataframe = pd.read_csv('data/train.csv')
dataframe['Class'] = dataframe['Class'].apply(lambda x: class_age_to_num[x])
dataframe.to_csv('data/encoded_train.csv', index = False)
