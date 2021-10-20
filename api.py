from typing import Optional, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

from codenames_board import CodenamesBoard
from word_association import similar_words, read_codenames_board
from words import WORDS

app = Flask(__name__)
CORS(app)


def parse_query_list(key: str) -> list[str]:
    result = request.args.get(key)
    return result.split(" ") if result else []


def parse_query_param(key: str, default: Optional[Any] = None) -> Any:
    result = request.args.get(key)
    return result or default


@app.route("/")
def home_route():
    return "<p>CodeNamer api</p>"


@app.route("/clues")
def get_clues():
    positive = parse_query_list("positive")
    negative = parse_query_list("negative")
    neutral = parse_query_list("neutral")
    assassin = parse_query_param("assassin")
    n = parse_query_param("n", 10)
    try:
        board = CodenamesBoard(positive, negative, neutral, assassin)
    except ValueError:
        return {"error": "need to specify at least one word in query string"}
    return read_codenames_board(board, int(n))


@app.route("/words/<word>")
def get_similar_words(word):
    num = parse_query_param("num", 10)
    # can specify "valid=false" in query string to return top n clues regardless of validity for codenames
    wants_valid_clues = parse_query_param("valid", "true").lower() == "true"
    return similar_words(word, int(num), wants_valid_clues)


@app.route("/words")
def get_words():
    return jsonify(WORDS)
