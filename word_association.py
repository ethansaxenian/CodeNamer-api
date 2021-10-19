from gensim.models import KeyedVectors


def load_model() -> KeyedVectors:
    return KeyedVectors.load(f'models/wiki-vectors')


def is_valid_clue(clue: str, matcher: str) -> bool:
    return clue.isalpha() and (matcher.lower() not in clue.lower()) and (clue.lower() not in matcher.lower())


def get_similar_words(word: str, number: int = 5, wants_valid_clues: bool = True) -> dict[str, float]:
    model = load_model()

    if not wants_valid_clues:
        clues = model.most_similar(word, topn=number)
        return {new_word.lower(): score for new_word, score in clues}

    temp = number
    while True:
        clues = model.most_similar(word, topn=temp)
        if not wants_valid_clues:
            return {new_word.lower(): score for new_word, score in clues}
        parsed_words = [(clue, score) for (clue, score) in clues if is_valid_clue(clue, word)]
        if len(parsed_words) == number:
            break
        temp += 1
    return {new_word.lower(): score for new_word, score in parsed_words}
