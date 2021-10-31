import itertools
from dataclasses import dataclass

from gensim.models import KeyedVectors

from codenames_board import CodenamesBoard
from models.fasttext_missing_words import fasttext_missing_words
from models.google_news_missing_words import google_news_missing_words

WordScoresDict = dict[str, float]


@dataclass
class Clue:
    word: str = None
    score: float = -1
    cards: list[str] = None


class NLPModel:
    def __init__(self):
        # possible models are fasttext-wiki-news-subwords-300 and word2vec-google-news-300
        self.model: KeyedVectors = KeyedVectors.load(f'models/word2vec-google-news-300')
        self.model.sort_by_descending_frequency()
        for key, vector in google_news_missing_words.items():
            self.model.add_vector(key, vector)

    def generate_similar_words(self, positive: list[str], negative: list[str], num: int):
        """given a Codenames board, returns the most similar words, regardless of validity"""
        return self.model.most_similar(positive=positive, negative=negative, topn=num)

    def smaller_model(self, board: CodenamesBoard, color: str, topn: int = 10) -> KeyedVectors:
        available_clues = self.model.most_similar(positive=board.positive(color), topn=topn)
        return self.model.vectors_for_all([w for w, s in available_clues] + board.board())

    def generate_valid_clues(self, board: CodenamesBoard, num: int, color: str) -> list[Clue]:
        """given a Codenames board, returns the most similar valid clues"""
        assert color in ("red", "blue")

        results = []

        smaller_model = self.smaller_model(board, color, 10000)

        for i in range(2, 5):
            for positive_group in itertools.combinations(board.positive(color), i):

                print(positive_group)
                temp = num
                valid_clues = []

                while len(valid_clues) != num:
                    words = smaller_model.most_similar(positive=positive_group, negative=board.negative(color), topn=temp)

                    for (word, score) in words:
                        if not board.is_valid_clue(word):
                            pass
                        else:
                            new_clue = Clue(word, score, [w.lower().replace("_", " ") for w in positive_group])
                            valid_clues.append(new_clue)

                        if len(valid_clues) == num:
                            results.extend(valid_clues)
                            break

                    temp += 1

        return sorted(results, key=lambda result: result.score, reverse=True)
