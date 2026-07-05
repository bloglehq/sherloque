from nltk import word_tokenize, WordNetLemmatizer

ALLOWED_TABLE_FIELDS = {
    "document": ["url", "title"],
    "token": ["token"],
    "token_position": ["doc_id", "token_id", "position"],
    "term_doc_stats": ["token_id", "doc_id", "tf"],
    "link": ["from_doc_id", "to_doc_id"],
}


def preprocess_text(text: str) -> list[str]:
    """Lowercase, tokenize, drop non-alphanumeric tokens, and lemmatize."""
    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(tok)
        for tok in word_tokenize(text.lower())
        if any(ch.isalnum() for ch in tok)
    ]


__all__ = [
    "preprocess_text",
    "ALLOWED_TABLE_FIELDS",
]
