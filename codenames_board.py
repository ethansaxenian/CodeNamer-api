from typing import Optional


class CodenamesBoard:
    def __init__(self, red: Optional[list[str]] = None, blue: Optional[list[str]] = None,
                 tan: Optional[list[str]] = None, black: Optional[str] = None):
        self.red = red or []
        self.blue = blue or []
        self.neutral = tan or []
        self.assassin = black

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
        words = []
        if color == "red":
            words.extend(self.blue)
        elif color == "blue":
            words.extend(self.red)
        else:
            raise ValueError("color must be 'red' or 'blue'")

        return words + self.neutral + ([self.assassin] if self.has_assassin() else [])

    def is_valid_clue(self, clue: str) -> bool:
        if not clue.isalpha():
            return False
        if any(word.lower() in clue.lower() for word in self.board()):
            return False
        if any(clue.lower() in word.lower() for word in self.board()):
            return False
        return True
