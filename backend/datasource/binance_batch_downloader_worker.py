import logging
import time
import traceback
from threading import Thread
import json

import pytz
import requests
from datetime import datetime
from datasource.providers import DataSourceProviders
from logger.logger_builder import LoggerBuilder
from schema.bucketed_trade_data import BucketedTradeData
from storage.storage_layer import StorageLayer


class BinanceBatchDownloaderWorker(Thread):
    __BINANCE_API__ = 'https://api.binance.com'
    __TRADE_ENDPOINT__ = f'{__BINANCE_API__}/api/v3/klines'

    def __init__(self, name: str, provider: DataSourceProviders, bucketed_trade_data: BucketedTradeData,
                 storage: StorageLayer, logger: logging.Logger = None):
        super().__init__(name=name)
        self.provider = provider
        self.bucketed_trade_data = bucketed_trade_data
        self.storage = storage
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()

    def run(self):
        self.logger.info(f'Downloading data for {self.bucketed_trade_data}')

        sleep = 1
        good = 0
        while True:
            try:
                time.sleep(sleep)
                response = requests.get(url=self.__TRADE_ENDPOINT__,
                                        params={
                                            "symbol": self.bucketed_trade_data.symbol.upper(),
                                            "interval": self.bucketed_trade_data.bin_size,
                                            "limit": 1000,
                                            "startTime": self.bucketed_trade_data.start,
                                        })
                self.logger.info(response.status_code)

                if response.status_code == 429:
                    sleep = min(16, sleep * 2)
                    good = 0
                    self.logger.warning(f'Slowing down...:  {sleep} secs')
                elif response.status_code != 200:
                    raise Exception(f'Status Code: {response.status_code}. Content: {response.content}')

                results = json.loads(response.content)
                results = self._get_results_struct(results=results)

                self.logger.info(f'Download {len(results)} results (after seep {sleep} secs )')
                self.bucketed_trade_data = self.storage.save(provider=self.provider,
                                                             symbol=self.bucketed_trade_data.symbol,
                                                             bin_size=self.bucketed_trade_data.bin_size,
                                                             results=results,
                                                             offset=results[-1]['open_time'])
                self.bucketed_trade_data.start = results[-1]['open_time']
                self.logger.info(f'Next batch {self.bucketed_trade_data}')
                good += 1
                if good == 2:
                    sleep = max(1, sleep // 2)
                    self.logger.info(f'Increasing downloading speed: {sleep} secs')

                if results[0]['open_time'] == results[-1]['open_time']:
                    sleep = 60
                    self.logger.info(f'Slowing down since we got no results: {sleep} secs')

            except Exception as e:
                self.logger.error(f'{e} -> Fatal Error exiting')
                self.logger.error(f'StackTrace\n{traceback.format_exc()}')
                return -1

    def _get_results_struct(self, results):
        return [
            {
                'timestamp': datetime.fromtimestamp(r[0] / 1000).astimezone(pytz.UTC),
                'symbol': self.bucketed_trade_data.symbol,
                'open_time': int(r[0]),
                'open': float(r[1]),
                'high': float(r[2]),
                'low': float(r[3]),
                'close': float(r[4]),
                'volume': float(r[5]),
                'close_time': int(r[6]),
                'quote_asset_volume': float(r[7]),
                'trades': float(r[8]),
                'taker_buy_base_asset_volume': float(r[9]),
                'taker_buy_quote_asset_volume': float(r[10]),
                'ignore': float(r[11]),
            } for r in results]


if __name__ == '__main__':
    symbol = 'ETHBTC'
    BinanceBatchDownloaderWorker(name='binance-test',
                                 storage=None,
                                 bucketed_trade_data=BucketedTradeData(
                                     symbol=symbol, bin_size='1d', start=0)).run()
