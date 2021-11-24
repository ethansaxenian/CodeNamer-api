from typing import Optional

from nltk.stem.porter import PorterStemmer

from models.google_news_missing_words import google_news_preprocessing_dict


class CodenamesBoard:
    def __init__(self, red: Optional[list[str]] = None, blue: Optional[list[str]] = None,
                 tan: Optional[list[str]] = None, black: Optional[str] = None):
        self.red = [google_news_preprocessing_dict.get(word, word) for word in red] if red is not None else []
        self.blue = [google_news_preprocessing_dict.get(word, word) for word in blue] if blue is not None else []
        self.neutral = [google_news_preprocessing_dict.get(word, word) for word in tan] if tan is not None else []
        self.assassin = google_news_preprocessing_dict.get(black, black)

        self.ps = PorterStemmer()
        # don't want to compute stems for the board words multiple times
        self.board_stems = set(self.ps.stem(word) for word in self.board())
        # don't want to compute stems for the same clue multiple times
        self.saved_stems = {}

    def __repr__(self):
        return f"{self.red}\n{self.blue}\n{self.neutral}\n{self.assassin}"

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
        lowercase_clue = clue.lower()

        if not lowercase_clue.isalpha():
            return False

        for word in self.board():
            if word.lower() in lowercase_clue:
                return False
            if lowercase_clue in word.lower():
                return False

        try:
            if self.saved_stems[lowercase_clue] in self.board_stems:
                return False
        except KeyError:
            self.saved_stems[lowercase_clue] = self.ps.stem(lowercase_clue)
            if self.saved_stems[lowercase_clue] in self.board_stems:
                return False

        return True
