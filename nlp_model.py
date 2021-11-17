import itertools
from collections import defaultdict
from dataclasses import dataclass

from gensim.models import KeyedVectors

from codenames_board import CodenamesBoard
from models.fasttext_missing_words import fasttext_missing_words
from models.google_news_missing_words import google_news_missing_words


@dataclass
class Clue:
    word: str = None
    score: float = -1
    cards: list[str] = None


class NLPModel:
    def __init__(self, model_name: str = "word2vec-google-news-300"):
        # possible models are fasttext-wiki-news-subwords-300 and word2vec-google-news-300
        self.model: KeyedVectors = KeyedVectors.load(f'models/{model_name}')
        self.model.sort_by_descending_frequency()

        if model_name == "word2vec-google-news-300":
            missing_words = google_news_missing_words
        elif model_name == "fasttext-wiki-news-subwords-300":
            missing_words = fasttext_missing_words
        else:
            raise ValueError("Invalid model name")

        self.model.add_vectors(list(missing_words.keys()), list(missing_words.values()))

        # controls how many valid clues the algorithm will generate for each combination
        self.num_valid_clues_per_word_group = 100

        # controls how many clues of each size that will be returned in the response object
        self.clues_per_size_to_return = 5

    def smaller_model(self, board: CodenamesBoard, color: str, topn: int = 10) -> KeyedVectors:
        available_clues = self.model.most_similar(positive=board.positive(color), topn=topn)
        return self.model.vectors_for_all([w for w, s in available_clues] + board.board())

    def generate_valid_clues(self, board: CodenamesBoard, color: str) -> dict[int, list[Clue]]:
        assert color in ("red", "blue")

        smaller_model = self.smaller_model(board, color, 10000)

        results_by_number = defaultdict(list)

        for i in [2, 3, 4]:
            for positive_group in itertools.combinations(board.positive(color), i):

                temp = self.num_valid_clues_per_word_group
                valid_clues = []

                while len(valid_clues) != self.num_valid_clues_per_word_group:
                    words = smaller_model.most_similar(positive=positive_group, negative=board.negative(color), topn=temp)

                    for (word, score) in words:
                        if board.is_valid_clue(word):
                            # if there is an assassin, make sure that any clues are at most orthogonal to it
                            if not board.has_assassin() or smaller_model.similarity(word, board.assassin) <= 0:
                                new_clue = Clue(word.lower(), score, [w.lower().replace("_", " ") for w in positive_group])
                                valid_clues.append(new_clue)

                        if len(valid_clues) == self.num_valid_clues_per_word_group:
                            results_by_number[i].extend(valid_clues)
                            break

                    temp += 1

        unique_clues_by_number = defaultdict(list)

        for i in results_by_number.keys():

            all_clues = set(clue.word for clue in results_by_number[i])
            temp_unique_clues = []

            for word in all_clues:
                same_clues = [clue for clue in results_by_number[i] if clue.word == word]
                best_clue = max(same_clues, key=lambda clue: clue.score)
                temp_unique_clues.append(best_clue)

            sorted_temp_unique_clues = sorted(temp_unique_clues, key=lambda clue: clue.score, reverse=True)
            unique_clues_by_number[i].extend(sorted_temp_unique_clues[:self.clues_per_size_to_return])

        return unique_clues_by_number
