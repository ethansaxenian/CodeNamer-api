from typing import Optional, Any, Type

import markdown
import markdown.extensions.fenced_code
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

from codenames_board import CodenamesBoard
from nlp_model import NLPModel
from words import WORDS

app = Flask(__name__)
CORS(app)


def parse_query_list(key: str) -> list[str]:
    result = request.args.get(key)
    return result.split(" ") if result else []


def parse_query_param(key: str, default: Optional[Any] = None, return_type: Type = str) -> Any:
    return request.args.get(key, default=default, type=return_type)


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404


@app.route("/")
def readme():
    readme_file = open("README.md", "r")
    markdown_str = markdown.markdown(readme_file.read(), extensions=["fenced_code"])
    return markdown_str


@app.route("/clues")
def get_clues():
    positive = parse_query_list("positive")
    negative = parse_query_list("negative")
    neutral = parse_query_list("neutral")
    assassin = parse_query_param("assassin")
    num = parse_query_param("num", 10, int)
    board = CodenamesBoard(positive, negative, neutral, assassin)
    print(board)
    if not board.board():
        abort(404, description="missing required query parameter")
    model = NLPModel()
    return model.generate_valid_clues(board, num)


@app.route("/clues/<word>")
def get_similar_words(word):
    num = parse_query_param("num", 10, int)
    model = NLPModel()
    board = CodenamesBoard(positive=[word])
    return model.generate_valid_clues(board, num)


@app.route("/words")
def get_words():
    return jsonify(WORDS)
