import inspect
from functools import wraps
from typing import Callable

from nltk import word_tokenize, WordNetLemmatizer
from sqlalchemy.ext.asyncio.engine import AsyncEngine

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
