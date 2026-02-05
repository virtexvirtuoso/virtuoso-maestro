from typing import List, Dict

from rethinkdb import RethinkDB

from datasource.providers import DataSourceProviders
from storage.storage_layer import StorageLayer
from schema.bucketed_trade_data import BucketedTradeData


class RethinkDbStorageLayer(StorageLayer):
    __METADATA_TABLE__ = 'trade_metadata'

    def __init__(self, host: str, port: int, db: str):
        super().__init__()
        self.host = host
        self.port = port
        self.db = db
        self._create_db_if_not_exists(host=host, port=port, db=db)
        self.connection = None
        self._create_table_if_not_exists(table_name=self.__METADATA_TABLE__, primary_key='table_name')

    def spawn(self):
        return RethinkDbStorageLayer(host=self.host, port=self.port, db=self.db)

    def get_connection(self):
        if self.connection is None or not self.connection.is_open():
            r = RethinkDB()
            self.logger.warning(f'Reopening connection to Rethinkdb')
            self.connection = r.connect(host=self.host, port=self.port, db=self.db)

        return self.connection

    def add(self, provider: DataSourceProviders, symbol: str, bin_size: str):
        table_name = self._get_table_name(provider=provider, symbol=symbol, bin_size=bin_size)
        self._create_table_if_not_exists(table_name=table_name, primary_key='timestamp')
        self._update_metadata(table_name=table_name,
                              provider=provider,
                              symbol=symbol,
                              bin_size=bin_size)

    def bucketed_trade_data_list(self, provider: DataSourceProviders) -> List[BucketedTradeData]:
        r = RethinkDB()
        metadata = (r.table(self.__METADATA_TABLE__)
                     .filter(lambda t: t['provider'] == provider.value)
                     .run(self.get_connection()))

        for m in metadata:
            yield BucketedTradeData(symbol=m.get('symbol'),
                                    bin_size=m.get('bin_size'),
                                    start=m.get('start'))

    def save(self, provider: DataSourceProviders, symbol: str, bin_size: str, offset: int, results: List[Dict]) -> BucketedTradeData:
        table_name = self._get_table_name(provider=provider, symbol=symbol, bin_size=bin_size)
        r = RethinkDB()
        table_meta = r.table(self.__METADATA_TABLE__).get(table_name).run(self.get_connection())
        self._append_records(table_name, results)
        r.table(self.__METADATA_TABLE__).get(table_name).update({'start': offset}).run(self.get_connection())
        return BucketedTradeData(symbol=table_meta.get('symbol'),
                                 bin_size=table_meta.get('bin_size'),
                                 start=offset)

    def _get_table_name(self, provider: DataSourceProviders, symbol: str, bin_size: str):
        table_name = f'trade_{provider.value}_{symbol.lower()}_{bin_size}'
        return table_name

    def size(self) -> int:
        r = RethinkDB()
        return r.table(self.__METADATA_TABLE__).count().run(self.get_connection())

    def _create_table_if_not_exists(self, table_name: str, primary_key: str = None) -> bool:
        r = RethinkDB()
        tables = set(r.table_list().run(self.get_connection()))
        if table_name in tables:
            self.logger.warning(f'table {table_name} is already present, data will be appended')
            return False
        else:
            self.logger.warning(f'table {table_name} does not exist, creating it')
            r.table_create(table_name, primary_key=primary_key).run(self.get_connection())
            return True

    def _update_metadata(self, table_name: str, provider: DataSourceProviders, symbol: str, bin_size: str):
        r = RethinkDB()
        r.table(self.__METADATA_TABLE__).wait().run(self.get_connection())
        record = r.table(self.__METADATA_TABLE__).get(table_name).run(self.get_connection())
        if record is not None:
            self.logger.warning(f'table {table_name} is already present, nothing to do')
        else:
            self.logger.warning(f'Adding metadata for table {table_name}')
            r.table(self.__METADATA_TABLE__).insert({
                'table_name': table_name,
                'provider': provider.value,
                'symbol': symbol,
                'bin_size': bin_size,
                'start': 0,
            }, conflict="replace").run(self.get_connection())

    def _create_db_if_not_exists(self, host: str, port: int, db: str):
        r = RethinkDB()
        con = r.connect(host=host, port=port)
        try:
            dbs = set(r.db_list().run(con))
            if db not in dbs:
                self.logger.warning(f'db {db} does not exist, creating it')
                r.db_create(db).run(con)
            else:
                self.logger.warning(f'db {db} already exists, great!')
        finally:
            con.close()

    def _append_records(self, table_name: str, results: List[Dict]):
        r = RethinkDB()
        status = r.table(table_name).insert(results, conflict="replace").run(self.get_connection())
        self.logger.info(f'Appended {len(results)} records. {status}')
