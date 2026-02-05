from __future__ import annotations

import logging
import sys
from logging import Logger


class LoggerBuilder:

    def __init__(self, name: str):
        self.name = name
        self.level = logging.INFO
        self.format = '%(asctime)s (%(threadName)-2s) %(levelname)s %(message)s'
        self.stream = sys.stdout

    def with_level(self, level: int) -> LoggerBuilder:
        self.level = level
        return self

    def with_format(self, format: str) -> LoggerBuilder:
        self.format = format
        return self

    def with_stream(self, stream: str) -> LoggerBuilder:
        self.stream = stream
        return self

    def build(self) -> Logger:
        if not self.name:
            raise Exception(f'You should provide a name for your logger')
        logger = logging.getLogger(name=self.name)
        logger.setLevel(level=self.level)
        if len(logger.handlers) == 0:
            hdlr = logging.StreamHandler(stream=sys.stdout)
            hdlr.setLevel(self.level)
            hdlr.setFormatter(logging.Formatter(fmt=self.format))
            logger.addHandler(hdlr=hdlr)

        return logger
