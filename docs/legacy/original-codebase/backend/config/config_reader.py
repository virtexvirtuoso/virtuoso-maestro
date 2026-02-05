from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import List, Dict

import yaml

from datasource.providers import DataSourceProviders
from logger.logger_builder import LoggerBuilder


class ConfigReader:

    def __init__(self, logger: Logger = None):
        self.config = None
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()

    def read(self, path: Path) -> ConfigReader:
        self.logger.info(f'Reading from {path}')
        with open(str(path.absolute()), 'r') as f:
            self.config = yaml.safe_load(f)['config']
        return self

    def get_rethinkdb_config(self) -> RethinkDbConfig:
        return RethinkDbConfig(**self.config['rethinkdb'])

    def get_datasource_config(self, provider: DataSourceProviders) -> DataSourceConfig:
        bitmex = self.config['datasource'][provider.value.lower()]['symbols']
        symbols = {s: DataSourceTradeConfig(**{**{'symbol': s}, **v}) for s, v in bitmex.items()}
        return DataSourceConfig(symbols=symbols)

    def get_restapi_config(self):
        return RestAPIConfig(**self.config['restapi'])

    def get_walkforward_config(self):
        return WalkForwardConfig(**self.config['optimization']['kind']['walkforward'])

    def get_optimization_output_config(self):
        return OptimizationOutputConfig(**self.config['optimization']['output'])


@dataclass
class OptimizationOutputConfig:
    results: str
    progress: str


@dataclass
class RethinkDbConfig:
    host: str
    port: int
    db: str


@dataclass
class DataSourceConfig:
    symbols: Dict[str, DataSourceTradeConfig]


@dataclass
class DataSourceTradeConfig:
    symbol: str
    bin_size: List[str]


@dataclass
class RestAPIConfig:
    host: str
    port: int
    debug: bool


@dataclass
class WalkForwardConfig:
    num_splits: int
