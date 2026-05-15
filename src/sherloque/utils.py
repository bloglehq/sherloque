import inspect
from functools import wraps
from typing import Callable

from nltk import word_tokenize, WordNetLemmatizer
from sqlalchemy.ext.asyncio.engine import AsyncEngine

ALLOWED_TABLE_FIELDS = {
    "document": ["url", "title"],
    "token": ["token"],
    "token_position": ["doc_id", "token_id", "position"],
    "term_doc_stats": ["token_id", "doc_id", "tf"],
    "link": ["from_doc_id", "to_doc_id"],
}


async def preprocess_text(text: str) -> list[str]:
    """Lowercase, tokenize, drop non-alphanumeric tokens, and lemmatize."""
    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(tok)
        for tok in word_tokenize(text.lower())
        if any(ch.isalnum() for ch in tok)
    ]


def insert_async_conn(engine: AsyncEngine):
    # decorator factory
    def decorator(func: Callable):
        params = list(inspect.signature(func).parameters.values())
        first_param_name = params[0].name if params else None
        expects_bound_first = first_param_name in {"self", "cls"}

        @wraps(func)
        async def create_cursor(*args, **kwargs):
            async with engine.connect() as conn:
                if expects_bound_first and args:
                    bound_first_arg, remaining_args = args[0], args[1:]
                    result = await func(bound_first_arg, conn, *remaining_args, **kwargs)
                else:
                    result = await func(conn, *args, **kwargs)
            return result

        return create_cursor

    return decorator


__all__ = [
    "preprocess_text",
    "insert_async_conn",
]
