import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#fix random seed for consistency
np.random.seed(7)

with open('data/word_vectors_subset.json', 'r') as fp:
    word_vectors = json.load(fp)

print(len(word_vectors))

data = pd.read_csv("data/train_subset.csv", encoding='utf-8')
#turn string to lowercase
data['question_text'] = data['question_text'].str.lower()
#remove all non-aplhanumberic characters
data['question_text'] = data['question_text'].str.replace('[^a-zA-Z0-9\s]', '')
#split words into own columns
split_data = data['question_text'].str.split(' ', expand=True)
split_data['target'] = data['target']

print("Training data ready")

# none = [0] * 300
# for row in split_data:
#     for column in row:
#         if column is None:
#             column = none
#         else:
#             column = word_vectors[column]

# print(data.head())
print(split_data.head())

none = [0] * 300
row_nbr = 0
#convert words in columns to vectors
row_nbr = 0
for row in split_data.itertuples():
    print(row)
    col_nbr = 1
    for col in row:
        if col_nbr < 45:
            split_data.iat[row_nbr, col_nbr] = word_vectors.get(split_data.iat[row_nbr, col_nbr], none)
            col_nbr += 1
    row_nbr += 1
print(split_data.head())

#divide data to training and testing sets
train, test = train_test_split(split_data, test_size=0.2)

# print("Train: %s,test: %s" % (len(train), len(test)))
# print(train)
# print(test)


# X_train = train[0:42]
# print(X_train)
# y_train = train['target']
# print(y_train)
#
# model = Sequential()
# #input size is 44 x 300 = 13200
# model.add(Dense(50, activation='relu', input_dim=45))
# model.add(Dense(1, activation='relu'))
#
# model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
#
# model.fit(X_train, y_train, epochs=5, batch_size=10)
#
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

