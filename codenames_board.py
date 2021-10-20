from typing import Optional


class CodenamesBoard:
    def __init__(self, positive: Optional[list[str]] = None, negative: Optional[list[str]] = None,
                 neutral: Optional[list[str]] = None, assassin: Optional[str] = None):
        self.positive = positive or []
        self.negative = negative or []
        self.neutral = neutral or []
        self.assassin = assassin
        if not bool(self.board()):
            raise ValueError("Codenames Board must contain at least one word")

    def board(self) -> list[str]:
        return self.positive + self.negative + self.neutral + ([self.assassin] if self.has_assassin() else [])

    def has_assassin(self) -> bool:
        return self.assassin is not None

    def negative_associated(self) -> list[str]:
        return self.negative + self.neutral + ([self.assassin] if self.has_assassin() else [])
