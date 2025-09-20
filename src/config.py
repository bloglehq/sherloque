import os
from functools import lru_cache, wraps
from pathlib import Path
from typing import Callable

import psycopg
from dotenv import load_dotenv
from pydantic.v1 import BaseSettings


class DefaultSettings(BaseSettings):
    DB_URL: str = os.getenv(
        "DB_URL",
        "postgresql://postgres:postgres@localhost:5432/sherloque",
    )


class Settings(DefaultSettings):
    pass


@lru_cache()
def get_settings() -> Settings:
    env_file = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_file)
    return Settings()


def manage_db_cursor(commit: bool = False):
    # decorator factory
    def decorator(func: Callable):
        is_class_func: bool = len(func.__qualname__.split('.')) > 1

        @wraps(func)
        async def create_cursor(*args, **kwargs):
            async with await psycopg.AsyncConnection.connect(get_settings().DB_URL) as conn:
                async with conn.cursor() as cursor:
                    # call the original function with the cursor and commit if asked
                    if is_class_func:
                        # inject cursor after `self`
                        self_instance, other_args = args[0], args[1:]
                        result = await func(self_instance, cursor, *other_args, **kwargs)
                    else:
                        result = await func(cursor, *args, **kwargs)
                    if commit:
                        await conn.commit()
            return result
        return create_cursor
    return decorator


__all__ = [
    "DefaultSettings",
    "Settings",
    "get_settings",
    "manage_db_cursor"
]
