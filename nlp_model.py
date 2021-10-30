import itertools
from dataclasses import dataclass

from gensim.models import KeyedVectors

from codenames_board import CodenamesBoard

WordScoresDict = dict[str, float]


@dataclass
class Clue:
    word: str = None
    score: float = -1
    cards: list[str] = None


class NLPModel:
    def __init__(self):
        # possible models are fasttext-wiki-news-subwords-300 and word2vec-google-news-300
        self.model: KeyedVectors = KeyedVectors.load(f'models/fasttext-wiki-news-subwords-300')
        self.model.sort_by_descending_frequency()
        self.english_words = set(line.strip().lower() for line in open("english_words.txt"))

    def is_valid_english_word(self, word: str) -> bool:
        return word in self.english_words

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
                            new_clue = Clue(word.lower(), score, positive_group)
                            valid_clues.append(new_clue)

                        if len(valid_clues) == num:
                            results.extend(valid_clues)
                            break

                    temp += 1

        return sorted(results, key=lambda result: result.score, reverse=True)
