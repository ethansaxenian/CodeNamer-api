from typing import Optional

WordScoresDict = dict[str, float]


class CodenamesBoard:
    def __init__(self, positive: Optional[list[str]] = None, negative: Optional[list[str]] = None,
                 neutral: Optional[list[str]] = None, assassin: Optional[str] = None):
        self.positive = positive or []
        self.negative = negative or []
        self.neutral = neutral or []
        self.assassin = assassin

    def __repr__(self):
        return f"{self.positive}\n{self.negative}\n{self.neutral}\n{self.assassin}"

    def has_assassin(self) -> bool:
        return self.assassin is not None

    def board(self) -> list[str]:
        return self.positive + self.negative + self.neutral + ([self.assassin] if self.has_assassin() else [])

    def negative_associated(self) -> list[str]:
        return self.negative + self.neutral + ([self.assassin] if self.has_assassin() else [])

    def is_valid_clue(self, clue: str) -> bool:
        if not clue.isalpha():
            return False
        if any(word.lower() in clue.lower() for word in self.board()):
            return False
        if any(clue.lower() in word.lower() for word in self.board()):
            return False
        return True
