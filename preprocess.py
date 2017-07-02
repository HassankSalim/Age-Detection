import pandas as pd
from sklearn.cross_validation import train_test_split

class_age_to_num = {'YOUNG':0, 'MIDDLE':1, 'OLD':2}

dataframe = pd.read_csv('data/pre_train.csv')
dataframe['Class'] = dataframe['Class'].apply(lambda x: class_age_to_num[x])

features_list = ['ID']
target = ['Class']

def split_to_train_val():

    filenames = dataframe[features_list]
    labels = dataframe[target]
    train_data, train_labels, vali_data, vali_labels = train_test_split(filenames, labels, test_size = 0.01, random_state = 3)

    return train_data, train_labels, vali_data, vali_labels

train_data, vali_data, train_labels, vali_labels = split_to_train_val()

train = pd.concat([train_data, train_labels], axis=1)
vali = pd.concat([vali_data, vali_labels], axis=1)

train.to_csv('data/train_set.csv', header=False, index = False)
vali.to_csv('data/vali_set.csv', header=False, index = False)
