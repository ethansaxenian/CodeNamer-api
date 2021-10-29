import sys

import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors

from words import WORDS

MODEL_NAME = "word2vec-google-news-300"
KEYED_VECTORS_PATH = f"~/gensim-data/{MODEL_NAME}/{MODEL_NAME}.gz"

# api.load(MODEL_NAME)

#
model = gensim.models.KeyedVectors.load_word2vec_format(
    KEYED_VECTORS_PATH, binary=True, limit=80000, unicode_errors='ignore'
)

model.save(f"models/{MODEL_NAME}")

# model = KeyedVectors.load(f"models/{MODEL_NAME}")
#
#
# x = ['africa', 'alps', 'amazon', 'america', 'antarctica', 'atlantis', 'australia', 'aztec', 'beijing', 'berlin', 'bermuda', 'bugle', 'canada', 'centaur', 'czech', 'egypt', 'england', 'europe', 'france', 'germany', 'greece', 'himalayas', 'hollywood', 'ice cream', 'jupiter', 'leprechaun', 'loch ness', 'london', 'mexico', 'moscow', 'new york', 'olympus', 'platypus', 'robin', 'rome', 'saturn', 'scorpion', 'scuba diver', 'shakespeare', 'tokyo', 'undertaker', 'unicorn', 'washington']
#
# bad_words = []
#
# d = {i: i.capitalize() for i in x}
#
# # print(d)
#
# w = {'alps': 'Alps', 'antarctica': 'Antarctica', 'atlantis': 'Atlantis', 'aztec': 'Aztec', 'beijing': 'Beijing', 'berlin': 'Berlin', 'bermuda': 'Bermuda', 'centaur': 'Centaur', 'czech': 'Czech', 'himalayas': 'Himalayas', 'ice cream': 'Ice cream', 'jupiter': 'Jupiter', 'leprechaun': 'Leprechaun', 'loch ness': 'Loch ness', 'moscow': 'Moscow', 'new york': 'New york', 'olympus': 'Olympus', 'rome': 'Rome', 'saturn': 'Saturn', 'scuba diver': 'Scuba diver', 'shakespeare': 'Shakespeare', 'tokyo': 'Tokyo'}
#
#
# if len(sys.argv) > 1:
#     print(model.most_similar(sys.argv[1]))
# else:
#     for i in x:
#         try:
#             model.most_similar(d[i], topn=1)
#         except KeyError:
#             bad_words.append(i)
#
# print(bad_words)
