from gensim.models import KeyedVectors

from codenames_board import CodenamesBoard

WordScoresDict = dict[str, float]


class NLPModel:
    def __init__(self):
        self.model = KeyedVectors.load(f'models/saved-vectors')

    def generate_similar_words(self, board: CodenamesBoard, num: int, color: str):
        """given a Codenames board, returns the most similar words, regardless of validity"""
        assert color in ("red", "blue")

        return self.model.most_similar(positive=board.positive(color), negative=board.negative(color), topn=num)

    def generate_valid_clues(self, board: CodenamesBoard, num: int, color: str) -> WordScoresDict:
        """given a Codenames board, returns the most similar valid clues"""
        assert color in ("red", "blue")

        temp = num
        while True:
            words = self.generate_similar_words(board, temp, color)
            valid_clues = [(word, score) for (word, score) in words if board.is_valid_clue(word)]
            if len(valid_clues) == num:
                break
            temp += 1

        return {clue.lower(): score for clue, score in valid_clues}
