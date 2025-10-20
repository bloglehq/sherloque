from nltk import word_tokenize, WordNetLemmatizer

ALLOWED_TABLE_FIELDS = {
    "url_list": ["url"],
    "token_list": ["token"],
    "token_location": ["url_id", "token_id", "location"],
    "link": ["from_id", "to_id"],
    "link_tokens": ["token_id", "link_id"],
}


async def preprocess_text(text: str) -> list[str]:
    """Tokenize and stem the text using NLTK"""
    toks = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok) for tok in toks]
