from flask import Flask, request, jsonify
from flask_cors import CORS
from word_association import get_similar_words
from words import WORDS

app = Flask(__name__)
CORS(app)


@app.route("/")
def home_route():
    return "<p>CodeNamer api</p>"


@app.route("/clues/<word>")
@app.route("/clues/<word>/<num>")
def get_clues(word, num=100):
    # can specify "valid=false" in query string to return top n clues regardless of validity for codenames
    wants_valid_clues = not (request.args and request.args.get('valid').lower() == "false")
    return get_similar_words(word, int(num), wants_valid_clues)


@app.route("/words")
def get_words():
    return jsonify(WORDS)
