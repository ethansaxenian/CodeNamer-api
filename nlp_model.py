from gensim.models import KeyedVectors

from codenames_board import CodenamesBoard

WordScores = list[tuple[str, float]]
WordScoresDict = dict[str, float]


class NLPModel:
    def __init__(self):
        self.model = KeyedVectors.load(f'models/saved-vectors')

    def generate_similar_words(self, board: CodenamesBoard, num: int):
        """given a Codenames board, returns the most similar words, regardless of validity"""
        return self.model.most_similar(positive=board.positive, negative=board.negative_associated(), topn=num)

    def generate_valid_clues(self, board: CodenamesBoard, num: int) -> WordScoresDict:
        """given a Codenames board, returns the best valid clues"""
        temp = num
        while True:
            words = self.generate_similar_words(board, temp)
            valid_clues = [(word, score) for (word, score) in words if board.is_valid_clue(word)]
            if len(valid_clues) == num:
                break
            temp += 1
        return {clue.lower(): score for clue, score in valid_clues}
