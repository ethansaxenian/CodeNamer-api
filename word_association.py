from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader as api

MODEL_NAME = "glove-wiki-gigaword-300"
KEYED_VECTORS_PATH = f"~/gensim-data/{MODEL_NAME}/{MODEL_NAME}.gz"


def load_word_vectors() -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(KEYED_VECTORS_PATH, binary=True, limit=500000)


def get_similar_words(model: KeyedVectors, word: str, number: int = 5) -> list[tuple[str, float]]:
    return model.most_similar(word, topn=number)


def save_model(model: KeyedVectors, name: str):
    model.save(f'models/{name}')


def load_model() -> KeyedVectors:
    return KeyedVectors.load(f'models/wiki-vectors')


def is_valid_clue(clue: str, matcher: str) -> bool:
    return ("_" not in clue.lower()) and (matcher.lower() not in clue.lower()) and (clue.lower() not in matcher.lower())


if __name__ == '__main__':
    model = api.load("glove-wiki-gigaword-100")
    save_model(model, "wiki-vectors-100")
    # dataset = api.load("wiki-english-20171001")
    # model = Word2Vec(dataset)
    # model.wv.save("models/vectors.kv")
