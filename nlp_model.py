import itertools
from collections import defaultdict
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
        # board_vectors = [(word, self.model.get_vector(word)) for word in board.board()]
        return self.model.vectors_for_all([w for w, s in available_clues] + board.board())

    def generate_valid_clues(self, board: CodenamesBoard, num: int, color: str) -> list[Clue]:
        """given a Codenames board, returns the most similar valid clues"""
        assert color in ("red", "blue")

        word_scores = defaultdict(lambda: Clue())

        results = []

        smaller_model = self.smaller_model(board, color, 10000)

        for i in range(2, 5):
            print(i)
            for positive_group in itertools.combinations(board.positive(color), i):
                print(positive_group)
                temp = num
                valid_clues = []
                while True:
                    words = smaller_model.most_similar(positive=positive_group, negative=board.negative(color), topn=temp)

                    for (word, score) in words:
                        processed_word = word.lower()
                        if word_scores[processed_word].score >= score or not board.is_valid_clue(processed_word):
                            pass
                        else:
                            new_clue = Clue(processed_word, score, positive_group)
                            #     word_scores[processed_word] = new_clue
                            valid_clues.append(new_clue)

                    # print(valid_clues)

                    if len(valid_clues) == num:
                        break
                    temp += 1

                    for clue in valid_clues:
                        word_scores[clue.word] = clue
                        results.append(clue)


        # for i in range(2, 3):
        #     for positive_group in itertools.combinations(board.positive(color), i):
        #         temp = num
        #         valid_clues = []
        #         while True:
        #             words = sorted(self.generate_similar_words(positive_group, board.negative(color), temp),
        #                            key=lambda pair: -pair[1])
        #
        #             for (word, score) in words:
        #                 processed_word = word.lower()
        #                 if word_scores[processed_word].score >= score or not board.is_valid_clue(
        #                         processed_word):
        #                     pass
        #                 else:
        #                     new_clue = Clue(processed_word, score, positive_group)
        #                     #     word_scores[processed_word] = new_clue
        #                     valid_clues.append(new_clue)
        #
        #             print(valid_clues)
        #
        #             if len(valid_clues) == num:
        #                 break
        #             temp += 1
        #
        #             for clue in valid_clues:
        #                 word_scores[clue.word] = clue
        #                 results.append(clue)

        return sorted(word_scores.values(), key=lambda result: result.score, reverse=True)[:num]
