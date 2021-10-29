from typing import Optional

from words import WORDS


class CodenamesBoard:
    preprocessing_dict = {
        'alps': 'Alps',
        'antarctica': 'Antarctica',
        'atlantis': 'Atlantis',
        'aztec': 'Aztec',
        'beijing': 'Beijing',
        'berlin': 'Berlin',
        'bermuda': 'Bermuda',
        'centaur': 'Centaur',
        'czech': 'Czech',
        'himalayas': 'Himalayas',
        'ice cream': 'ice-cream',
        'jupiter': 'Jupiter',
        'leprechaun': 'elf',
        'loch ness': 'Loch',
        'moscow': 'Moscow',
        'new york': 'New-York',
        'olympus': 'Olympus',
        'rome': 'Rome',
        'saturn': 'Saturn',
        'scuba diver': 'diver',
        'shakespeare': 'Shakespeare',
        'tokyo': 'Tokyo'
    }

    def __init__(self, red: Optional[list[str]] = None, blue: Optional[list[str]] = None,
                 tan: Optional[list[str]] = None, black: Optional[str] = None):
        self.red = red or []
        self.blue = blue or []
        self.neutral = tan or []
        self.assassin = black
        # self.english_words = set(line.strip().lower() for line in open("english_words.txt"))
        self.english_words = set(WORDS)

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
        if not clue.isalpha():
            return False
        if any(word.lower() in clue.lower() for word in self.board()):
            return False
        if any(clue.lower() in word.lower() for word in self.board()):
            return False
        # if clue not in self.english_words:
        #     return False
        return True
