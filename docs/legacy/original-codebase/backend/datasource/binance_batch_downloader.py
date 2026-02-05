import logging
from threading import Thread
from typing import List

from datasource.binance_batch_downloader_worker import BinanceBatchDownloaderWorker
from datasource.providers import DataSourceProviders
from logger.logger_builder import LoggerBuilder
from storage.storage_layer import StorageLayer


class BinanceBatchDownloader:

    def __init__(self, storage: StorageLayer, logger: logging.Logger = None):
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()
        self.storage = storage
        self.provider = DataSourceProviders.BINANCE

    def register(self, symbol: str, bin_size: str):
        self.logger.info(f'Registering {symbol} with bin size {bin_size}')
        self.storage.add(symbol=symbol, bin_size=bin_size, provider=self.provider)
        return self

    def start(self):
        threads: List[Thread] = []
        for bucketed_trade_data in self.storage.bucketed_trade_data_list(provider=self.provider):
            t_name = f'downloader-{self.provider}-{bucketed_trade_data.symbol}-{bucketed_trade_data.bin_size}'
            t = BinanceBatchDownloaderWorker(
                name=t_name,
                provider=self.provider,
                bucketed_trade_data=bucketed_trade_data,
                storage=self.storage.spawn(),
                logger=LoggerBuilder(name=t_name).build()
            )
            t.start()

        for t in threads:
            t.join()
