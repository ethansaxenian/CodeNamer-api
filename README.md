# CodeNamer-api

A companion api designed to work alongside the [CodeNamer](https://github.com/ethansaxenian/CodeNamer) app that provides
various python scripts for processing images of Codenames boards and clue generation.

Visit the api online: [https://code-namer.herokuapp.com/](https://code-namer.herokuapp.com/)

### Installation and setup instructions:

Install tesseract using `homebrew`:
```
brew install tesseract
```

Create and enter a python virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the flask server:
```
flask run
```
