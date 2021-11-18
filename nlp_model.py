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


CluesResponse = dict[int, list[Clue]]


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
        self.num_valid_clues_per_word_group = 50

        # controls how many clues of each size that will be returned in the response object
        self.clues_per_size_to_return = 5

    def smaller_model(self, board: CodenamesBoard, color: str, size: int) -> KeyedVectors:
        """
        returns a model that only includes clues that are positively associated with one color of a Codenames board
        """
        available_clues = self.model.most_similar(positive=board.positive(color), topn=size)
        return self.model.vectors_for_all([w for w, s in available_clues] + board.board())

    def generate_valid_clues(self, board: CodenamesBoard, color: str) -> CluesResponse:
        """
        given a Codenames board and a color, returns an dict with the best clues for 2, 3, and 4 words
        """
        assert color in ("red", "blue")

        # create a much smaller model by only including words that are positively correlated with the good words
        # this will increase efficiency and remove the vast majority of words that don't make sense
        smaller_model = self.smaller_model(board, color, 10000)

        # sort results by clue size
        results_by_number = defaultdict(list)

        # keep track of all invalid clues so we aren't computing stems more than necessary (increases efficiency)
        invalid_clues = set()

        for i in [2, 3, 4]:
            for positive_group in itertools.combinations(board.positive(color), i):

                temp = self.num_valid_clues_per_word_group
                valid_clues = []

                # continue calculating more and more potential clues until we have enough
                while len(valid_clues) != self.num_valid_clues_per_word_group:

                    # unfortunately we have to generate the same words over and over, which reduces efficiency
                    # there is no way that I've found to skip over words that have already been generated
                    words = smaller_model.most_similar(positive=positive_group, negative=board.negative(color), topn=temp)

                    for (word, score) in words:
                        if word.lower() not in invalid_clues and board.is_valid_clue(word):
                            # if there is an assassin, make sure that any clues are at most orthogonal to it
                            if not board.has_assassin() or smaller_model.similarity(word, board.assassin) <= 0:
                                new_clue = Clue(word.lower(), score, [w.lower().replace("_", " ") for w in positive_group])
                                valid_clues.append(new_clue)
                        else:
                            invalid_clues.add(word.lower())

                        # once we've found enough clues, move on to the next group of words
                        if len(valid_clues) == self.num_valid_clues_per_word_group:
                            results_by_number[i].extend(valid_clues)
                            break

                    temp += 1

        return self.parse_clue_results(results_by_number)

    def parse_clue_results(self, results_by_number: CluesResponse) -> CluesResponse:
        """
        reduces <results_by_number> to the <clues_per_size_to_return> best clues for each clue size
        """
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
