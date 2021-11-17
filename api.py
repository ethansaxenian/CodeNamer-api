from typing import Optional, Any, Type

import markdown
import markdown.extensions.fenced_code
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import base64

from codenames_board import CodenamesBoard
from nlp_model import NLPModel
from color_recognition import colorCard
from text_recognition import gameBoard
from words import WORDS

app = Flask(__name__)
CORS(app)


def parse_query_list(key: str) -> list[str]:
    result = request.args.get(key)
    return result.split(" ") if result else []


def parse_query_param(key: str, default: Optional[Any] = None, return_type: Type = str) -> Any:
    return request.args.get(key, default=default, type=return_type)


def convert_and_save(b64_string):
    with open("imageToSave.jpeg", "wb") as fh:
        fh.write(base64.b64decode(b64_string))


@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400


@app.route("/")
def readme():
    readme_file = open("README.md", "r")
    markdown_str = markdown.markdown(readme_file.read(), extensions=["fenced_code"])
    return markdown_str


@app.route("/clues/<color>")
def get_clues(color):
    if not color.lower() in ("red", "blue"):
        abort(400, description="Invalid color. Must be 'red' or 'blue'")

    reds = parse_query_list("red")
    blues = parse_query_list("blue")
    tans = parse_query_list("tan")
    black = parse_query_param("black")

    board = CodenamesBoard(reds, blues, tans, black)

    if not board.board():
        abort(400, description="Missing required query parameter. "
                               "At least one of 'positive', 'negative', 'neutral', 'assassin' required.")
    model = NLPModel()
    return jsonify(model.generate_valid_clues(board, color.lower()))


@app.route("/words/<word>")
def get_clues_for_word(word):
    num = parse_query_param("num", 10, int)
    model = NLPModel()
    board = CodenamesBoard(red=[word])
    return model.generate_valid_clues(board, num, color="red")


@app.route("/colors", methods=['POST'])
def get_color_code():
    if request.data:
        card = colorCard()
        return jsonify(card.getColorCode(request.data))
    return jsonify("error")


@app.route("/gameboard", methods=['POST'])
def get_game_text():
    if request.data:
        board = gameBoard()
        print(jsonify(board.getGameText(request.data)))
        return jsonify(board.getGameText(request.data))
    return jsonify("error")


@app.route("/words")
def get_words():
    return jsonify(WORDS)
