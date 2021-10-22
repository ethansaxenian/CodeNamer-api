from gensim.models import KeyedVectors

from codenames_board import CodenamesBoard

WordScores = list[tuple[str, float]]
WordScoresDict = dict[str, float]


class NLPModel:
    def __init__(self):
        self.model = KeyedVectors.load(f'models/wiki-vectors')

    def similar_words(self, word: str, number: int = 5, wants_valid_clues: bool = True) -> WordScoresDict:
        if not wants_valid_clues:
            clues = self.model.most_similar(positive=[word], topn=number)
            return {new_word.lower(): score for new_word, score in clues}

        board = CodenamesBoard(positive=[word])

        valid_clues = self.get_n_valid_clues(board, number)
        return {new_word.lower(): score for new_word, score in valid_clues}

    def read_codenames_board(self, board: CodenamesBoard, n: int = 10) -> WordScoresDict:
        valid_clues = self.get_n_valid_clues(board, n)
        return {new_word.lower(): score for new_word, score in valid_clues}

    def get_clues(self, board: CodenamesBoard, n: int = 10) -> WordScores:
        return self.model.most_similar(positive=board.positive, negative=board.negative_associated(), topn=n)

    def get_n_valid_clues(self, board: CodenamesBoard, n: int = 10) -> WordScores:
        temp = n
        while True:
            clues = self.get_clues(board, temp)
            valid_clues = [(clue, score) for (clue, score) in clues if board.is_valid_clue(clue)]
            if len(valid_clues) == n:
                break
            temp += 1
        return valid_clues
