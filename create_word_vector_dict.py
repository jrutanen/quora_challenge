import json


word_vectors = {}

with open("data/wiki-news-300d-1M/wiki-news-300d-1M.vec", encoding='utf-8') as f:
    for line in f:
        if len(line) < 20:
            continue
        line = line.rstrip('\n')
        value = line.split(' ')
        key = str(value[0])
        value = value[1:]
        word_vectors[key] = value
with open('data/word_vectors.json', 'w') as fp:
    json.dump(word_vectors, fp)

print(len(word_vectors))
