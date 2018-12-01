import pandas as pd
import json
import re

target_0_max = 50
target_0 = 0
target_1_max = 50
target_1 = 0

# with open("data/train.csv", encoding='utf-8') as f:
#     with open('data/train_subset.csv', 'w', encoding='utf-8') as fw:
#         for line in f:
#             if line.find(",0\n") > -1:
#                 if target_0 <= target_0_max:
#                     target_0 += 1
#                     fw.write(line)
#             else:
#                 if target_1 <= target_1_max:
#                     target_0 += 1
#                     fw.write(line)
#             if target_1 + target_0 >= 102:
#                 break

data = pd.read_csv("data/train_subset.csv", encoding='utf-8')
#turn string to lowercase
data['question_text'] = data['question_text'].str.lower()
#remove all non-aplhanumberic characters
data['question_text'] = data['question_text'].str.replace('[^a-zA-Z0-9\s]', '')
#split words into own columns
split_data = data['question_text'].str.split(' ', expand=True)
print("Training data ready")

word_vectors = {}
#create a dictionary of the wiki word vector data
#with open("data/wiki_short.vec", encoding='utf-8') as f:
with open("data/wiki-news-300d-1M/wiki-news-300d-1M.vec", encoding='utf-8') as f:
    for line in f:
        if len(line) < 20:
            continue
        line = line.rstrip('\n')
        value = line.split(' ')
        key = str(value[0])
        value = value[1:]
        word_vectors[key] = value
print("Wiki word vector ready")

word_vector_subset = {}
none = [0] * 300
#Create dictionary of the words used in the data subset to reduce the size
#and processing time
for row in split_data.itertuples(index=True, name='Pandas'):
    for column in row:
        print(column)
        if column not in word_vector_subset:
            value = []
            key = column
            value = word_vectors.get(column, none)
            word_vector_subset.update({column: value})
            print("%s added to dictionary with value: %s" % (column, value))

print("Subset word vector ready")

#save word subset dictionary as a json file
with open('data/word_vectors_subset.json', 'w') as fp:
    json.dump(word_vector_subset, fp)
