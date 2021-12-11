import itertools
from collections import defaultdict
from dataclasses import dataclass

from gensim.models import KeyedVectors

from codenames_board import CodenamesBoard
from models.google_news_missing_words import google_news_missing_words


@dataclass
class Clue:
    word: str = None
    score: float = -1
    cards: list[str] = None


CluesResponse = dict[int, list[Clue]]


class NLPModel:
    def __init__(self, num_valid_clues_per_word_group: int = 10,
                 clues_per_size_to_return: int = 5, clue_buffer_size: int = 4096):
        self.model: KeyedVectors = KeyedVectors.load("models/word2vec-google-news-300")
        self.model.sort_by_descending_frequency()

        self.model.add_vectors(list(google_news_missing_words.keys()), list(google_news_missing_words.values()))

        # controls how many valid clues the algorithm will generate for each combination
        self.num_valid_clues_per_word_group = num_valid_clues_per_word_group

        # controls how many clues of each size that will be returned in the response object
        self.clues_per_size_to_return = clues_per_size_to_return

        # controls the number of potential clues to generate initially
        # 4096 should be more than enough for any request
        self.clue_buffer_size = clue_buffer_size

    def get_invalid_words(self, words: list[str]) -> list[str]:
        return [word for word in words if not self.model.has_index_for(word)]

    def smaller_model(self, board: CodenamesBoard, color: str, size: int = 10000) -> KeyedVectors:
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

        if not board.positive(color):
            return {}

        # create a much smaller model by only including words that are positively correlated with the good words
        # this will increase efficiency and remove the vast majority of words that don't make sense
        smaller_model = self.smaller_model(board, color)

        # sort results by clue size
        results_by_number = defaultdict(list)

        # keep track of all invalid clues so we aren't computing stems more than necessary (increases efficiency)
        invalid_clues = set()

        # weight assassin more strongly that other negative words
        weighted_negative_words = []
        for item in board.negative(color):
            if item == board.assassin:
                weight = -10.0
            elif item in board.opposite(color):
                weight = -3.0
            else:
                weight = -1.0
            weighted_negative_words.append((item, weight))

        for i in [2, 3, 4]:
            for positive_group in itertools.combinations(board.positive(color), i):

                potential_clue_buffer = smaller_model.most_similar(positive=positive_group,
                                                                   negative=weighted_negative_words,
                                                                   topn=self.clue_buffer_size)

                valid_clues = []

                for word, score in potential_clue_buffer:
                    if word.lower() not in invalid_clues and board.is_valid_clue(word):
                        # if there is an assassin, make sure that any clues are at most orthogonal to it
                        if not bool(board.assassin) or smaller_model.n_similarity([word], board.assassin) <= 0:
                            new_clue = Clue(word.lower(), score, [w.lower().replace("_", " ") for w in positive_group])
                            valid_clues.append(new_clue)
                    else:
                        invalid_clues.add(word.lower())

                    # once we've found enough clues, move on to the next group of words
                    if len(valid_clues) == self.num_valid_clues_per_word_group:
                        results_by_number[i].extend(valid_clues)
                        break

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
