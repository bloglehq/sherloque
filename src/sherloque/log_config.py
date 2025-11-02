import logging.config

def setup():
    LOGGING_CONFIG = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "level": "DEBUG",
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "loggers": {
            "sherloque": {
                "level": "DEBUG",
                "handlers": ["console", ],
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)

__all__ = [
    "setup",
]
