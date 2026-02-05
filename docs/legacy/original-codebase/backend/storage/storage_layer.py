from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import List, Dict

from datasource.providers import DataSourceProviders
from logger.logger_builder import LoggerBuilder
from schema.bucketed_trade_data import BucketedTradeData


class StorageLayer(ABC):

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()

    @abstractmethod
    def add(self, provider: DataSourceProviders, symbol: str, bin_size: str):
        raise NotImplementedError()

    @abstractmethod
    def bucketed_trade_data_list(self, provider: DataSourceProviders) -> List[BucketedTradeData]:
        raise NotImplementedError()

    @abstractmethod
    def save(self, provider: DataSourceProviders, symbol: str, bin_size: str,
             offset: int, results: List[Dict]) -> BucketedTradeData:
        raise NotImplementedError()

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def spawn(self) -> StorageLayer:
        raise NotImplementedError()
