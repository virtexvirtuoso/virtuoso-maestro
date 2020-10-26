import logging
import time
from threading import Thread
from typing import List, Tuple, Dict
import traceback
import bitmex
from bravado.exception import HTTPTooManyRequests
from bravado.requests_client import RequestsResponseAdapter

from datasource.providers import DataSourceProviders
from storage.storage_layer import StorageLayer
from logger.logger_builder import LoggerBuilder
from schema.bucketed_trade_data import BucketedTradeData


class BitMexBatchDownloaderWorker(Thread):

    def __init__(self, name: str, provider: DataSourceProviders, bucketed_trade_data: BucketedTradeData,
                 storage: StorageLayer, logger: logging.Logger = None):
        super().__init__(name=name)
        self.provider = provider
        self.bucketed_trade_data = bucketed_trade_data
        self.storage = storage
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()

    def run(self):
        client = bitmex.bitmex(test=True)
        self.logger.info(f'Downloading data for {self.bucketed_trade_data}')

        sleep = 1
        good = 0
        while True:
            try:
                time.sleep(sleep)
                response: Tuple[List[Dict], RequestsResponseAdapter] = client.Trade.Trade_getBucketed(
                    symbol=self.bucketed_trade_data.symbol.upper(),
                    binSize=self.bucketed_trade_data.bin_size,
                    start=self.bucketed_trade_data.start,
                    count=1_000
                ).result()

                results, request_response = response
                self.logger.info(request_response)

                self.logger.info(f'Download {len(results)} results (after seep {sleep} secs )')
                self.bucketed_trade_data = self.storage.save(provider=self.provider,
                                                             symbol=self.bucketed_trade_data.symbol,
                                                             bin_size=self.bucketed_trade_data.bin_size,
                                                             results=results,
                                                             offset=self.bucketed_trade_data.start + len(results))

                self.logger.info(f'Next batch {self.bucketed_trade_data}')
                good += 1
                if good == 2:
                    sleep = max(1, sleep // 2)
                    self.logger.info(f'Increasing downloading speed: {sleep} secs')

                if len(results) == 0:
                    sleep = 60
                    self.logger.info(f'Slowing down since we got no results: {sleep} secs')

            except HTTPTooManyRequests as e:
                sleep = min(16, sleep * 2)
                good = 0
                self.logger.warning(f'Slowing down...:  {sleep} secs')
            except Exception as e:
                self.logger.error(f'{e} -> Fatal Error exiting')
                self.logger.error(f'StackTrace\n{traceback.format_exc()}')
                return -1
