from gensim.models import KeyedVectors


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


def similar_words(word: str, number: int = 5, wants_valid_clues: bool = True) -> dict[str, float]:
    model = load_model()

    if not wants_valid_clues:
        clues = model.most_similar(word, topn=number)
        return {new_word.lower(): score for new_word, score in clues}

    temp = number
    while True:
        clues = model.most_similar(word, topn=temp)
        parsed_words = [(clue, score) for (clue, score) in clues if is_valid_clue(clue, [word])]
        if len(parsed_words) == number:
            break
        temp += 1
    return {new_word.lower(): score for new_word, score in parsed_words}


def read_codenames_board(positive: list[str], negative: list[str], neutral: list[str], assassin: list[str],
                         n: int = 10) -> dict[str, float]:
    model = load_model()

    temp = n
    while True:
        clues = model.most_similar(positive=positive, negative=negative + neutral + assassin, topn=temp)
        parsed_words = [(clue, score) for (clue, score) in clues if
                        is_valid_clue(clue, positive + negative + neutral + assassin)]
        if len(parsed_words) == n:
            break
        temp += 1
    return {new_word.lower(): score for new_word, score in parsed_words}
