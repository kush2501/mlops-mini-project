import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download only if not available
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load once (Fast)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# -------------------- Lower Case -------------------- #

def lower_case(text: str) -> str:
    return text.lower()


# -------------------- Remove Stop Words -------------------- #

def remove_stop_words(text: str) -> str:

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -------------------- Remove Numbers -------------------- #

def remove_numbers(text: str) -> str:

    return "".join(
        char
        for char in text
        if not char.isdigit()
    )


# -------------------- Remove Punctuation -------------------- #

def remove_punctuation(text: str) -> str:

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# -------------------- Remove URLs -------------------- #

def remove_urls(text: str) -> str:

    url_pattern = re.compile(
        r"https?://\S+|www\.\S+"
    )

    return url_pattern.sub("", text)


# -------------------- Lemmatization -------------------- #

def lemmatization(text: str) -> str:

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# -------------------- Complete Preprocessing -------------------- #

def preprocess_text(text: str) -> str:

    text = lower_case(text)

    text = remove_stop_words(text)

    text = remove_numbers(text)

    text = remove_punctuation(text)

    text = remove_urls(text)

    text = lemmatization(text)

    return text