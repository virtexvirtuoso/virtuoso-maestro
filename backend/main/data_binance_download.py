import os
from pathlib import Path

from datasource.binance_batch_downloader import BinanceBatchDownloader
from datasource.bitmex_batch_downloader import BitMexBatchDownloader
from datasource.providers import DataSourceProviders
from storage.rethinkdb_storage_layer import RethinkDbStorageLayer
from config.config_reader import ConfigReader

ROOT_DIR = Path(__file__).parents[1].absolute()


if __name__ == '__main__':

    CONFIG_FILE = os.environ.get('CONFIG_FILE', 'maestro-dev.yaml')

    path = ROOT_DIR.joinpath(CONFIG_FILE)
    config_reader = ConfigReader().read(path=path)

    db_config = config_reader.get_rethinkdb_config()
    storage = RethinkDbStorageLayer(host=db_config.host, port=db_config.port, db=db_config.db)
    binance_config = config_reader.get_datasource_config(provider=DataSourceProviders.BINANCE)
    binance_downloader = BinanceBatchDownloader(storage=storage)

    for symbol, sconf in binance_config.symbols.items():
        for bin_size in sconf.bin_size:
            binance_downloader.register(symbol=symbol, bin_size=bin_size)

    binance_downloader.start()














