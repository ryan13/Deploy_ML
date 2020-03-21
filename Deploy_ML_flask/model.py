# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('hiring.csv')

df['experience'].fillna(0, inplace=True)

df['test_score'].fillna(df['test_score'].mean(), inplace=True)

X = df.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = df.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all available data.

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#Fitting model with training data
lr.fit(X, y)

# Saving model to disk using pickle file
pickle.dump(lr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))