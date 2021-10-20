from typing import Callable

from gensim.models import KeyedVectors

from codenames_board import CodenamesBoard

WordScores = list[tuple[str, float]]
WordScoresDict = dict[str, float]


def load_model() -> KeyedVectors:
    return KeyedVectors.load(f'models/wiki-vectors')


def is_valid_clue(clue: str, board: list[str]) -> bool:
    if not clue.isalpha():
        return False
    if any(word.lower() in clue.lower() for word in board):
        return False
    if any(clue.lower() in word.lower() for word in board):
        return False
    return True


def similar_words(word: str, number: int = 5, wants_valid_clues: bool = True) -> WordScoresDict:
    model = load_model()

    if not wants_valid_clues:
        clues = model.most_similar(positive=[word], topn=number)
        return {new_word.lower(): score for new_word, score in clues}

    board = CodenamesBoard(positive=[word])

    valid_clues = get_n_valid_results(model, board, get_clues, number)
    return {new_word.lower(): score for new_word, score in valid_clues}


def read_codenames_board(board: CodenamesBoard, n: int = 10) -> WordScoresDict:
    model = load_model()
    valid_clues = get_n_valid_results(model, board, get_clues, n)
    return {new_word.lower(): score for new_word, score in valid_clues}


def get_clues(model: KeyedVectors, board: CodenamesBoard, n: int = 10) -> WordScores:
    return model.most_similar(positive=board.positive, negative=board.negative_associated(), topn=n)


def get_n_valid_results(model: KeyedVectors, board: CodenamesBoard,
                        get_words_fn: Callable[[KeyedVectors, CodenamesBoard, int], WordScores],
                        n: int = 10) -> WordScores:
    temp = n
    while True:
        clues = get_words_fn(model, board, temp)
        valid_clues = [(clue, score) for (clue, score) in clues if is_valid_clue(clue, board.board())]
        if len(valid_clues) == n:
            break
        temp += 1
    return valid_clues
