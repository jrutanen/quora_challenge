#import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#fix random seed for consistency
np.random.seed(7)

with open('data/word_vectors.json', 'r') as fp:
    word_vectors = json.load(fp)

print(len(word_vectors))

data = pd.read_csv("data/train.csv", encoding='utf-8')
#data = data['question_text'].str.lower()
#handle , . ! ' etc in some nice way
#data = data['question_text'].str.replace(r'![a-z]+', repl)

split_data = data['question_text'].str.split(' ', expand=True)

none = [0] * 300
for row in split_data:
    for column in row:
        if column is None:
            column = none
        else:
            column = word_vectors[column]

print(data.head())
print(split_data.head())


# train, test = train_test_split(data, test_size=0.2)
#
# print("Train: %s,test: %s" % (len(train), len(test)))
# print(data.head())
#
# X_train = train['question_text'].tolist()
# y_train = train['target'].tolist()

#model = Sequential()
#model.add(Dense(16, activation='relu', input_dim=1))
#model.add(Dense(1, activation='relu'))

#model.compile(loss='binary_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy'])

#model.fit(X_train, y_train, epochs=5, batch_size=10)

#scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

