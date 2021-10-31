from typing import Optional

from models.fasttext_missing_words import fasttext_preprocessing_dict
from models.google_news_missing_words import google_news_preprocessing_dict


class CodenamesBoard:
    # since the model is uncased, use this dict to convert input words to the correct form
    # note this dict is different depending on the model used
    preprocessing_dict = google_news_preprocessing_dict

    def __init__(self, red: Optional[list[str]] = None, blue: Optional[list[str]] = None,
                 tan: Optional[list[str]] = None, black: Optional[str] = None):
        self.red = [self.preprocess_word(word) for word in red] if red is not None else []
        self.blue = [self.preprocess_word(word) for word in blue] if blue is not None else []
        self.neutral = [self.preprocess_word(word) for word in tan] if tan is not None else []
        self.assassin = self.preprocess_word(black)

    def __repr__(self):
        return f"{self.red}\n{self.blue}\n{self.neutral}\n{self.assassin}"

    def preprocess_word(self, word: str) -> str:
        try:
            return self.preprocessing_dict[word]
        except KeyError:
            return word

    def has_assassin(self) -> bool:
        return self.assassin is not None

    def board(self) -> list[str]:
        return self.red + self.blue + self.neutral + ([self.assassin] if self.has_assassin() else [])

    def positive(self, color: str) -> list[str]:
        if color == "red":
            return self.red
        elif color == "blue":
            return self.blue
        else:
            raise ValueError("color must be 'red' or 'blue'")

    def negative(self, color: str) -> list[str]:
        negative_words = []
        if color == "red":
            negative_words.extend(self.blue)
        elif color == "blue":
            negative_words.extend(self.red)
        else:
            raise ValueError("color must be 'red' or 'blue'")

        return negative_words + self.neutral + ([self.assassin] if self.has_assassin() else [])

    def is_valid_clue(self, clue: str) -> bool:
        if not clue.isalpha():
            return False
        if any(word.lower() in clue.lower() for word in self.board()):
            return False
        if any(clue.lower() in word.lower() for word in self.board()):
            return False
        return True
