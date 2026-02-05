import hashlib
import os
from collections import defaultdict
from io import StringIO, BytesIO
from pathlib import Path
import pandas as pd
import pytz
from flask import Flask, request, make_response
from flask_cors import CORS
from flask_restful import Resource, Api, abort
from flask_gzip import Gzip
import xlsxwriter
import numpy as np
from rethinkdb import RethinkDB
from rethinkdb.errors import ReqlOpFailedError
import base64
from datasource.providers import DataSourceProviders
from strategy import __STRATEGY_CATALOG__
from config.config_reader import ConfigReader, RethinkDbConfig, OptimizationOutputConfig, WalkForwardConfig
from engine.backtesting_engine import BacktestingEngine
from engine.optimizer import Optimizer, OptimizationType
from engine.walk_forward_engine import WalkForwardEngine
from datetime import datetime, timedelta

ROOT_DIR = Path(__file__).parents[1].absolute()

app = Flask(__name__)
api = Api(app)
gzip = Gzip(app)
CORS(app)


class StrategyList(Resource):

    def get(self):
        return list(__STRATEGY_CATALOG__.keys())


class StrategyParams(Resource):

    def get(self, strategy: str):
        if strategy not in __STRATEGY_CATALOG__:
            return {}
        return __STRATEGY_CATALOG__.get(strategy).get_params()


class DataSourceProvider(Resource):

    def get(self):
        return list([e.value for e in DataSourceProviders])


class DataSourceProviderSymbol(Resource):

    def __init__(self, rethinkdb_config: RethinkDbConfig):
        self.r = RethinkDB()
        self.connection = self.r.connect(host=rethinkdb_config.host,
                                         port=rethinkdb_config.port,
                                         db=rethinkdb_config.db)

    def get(self, provider: str):
        return list(map(lambda x: x['symbol'].upper(), self.r
                        .table("trade_metadata")
                        .filter(lambda r: r['provider'] == provider.upper())
                        .pluck('symbol').distinct()
                        .run(self.connection)))


class DataSource(Resource):

    def __init__(self, rethinkdb_config: RethinkDbConfig):
        self.r = RethinkDB()
        self.connection = self.r.connect(host=rethinkdb_config.host,
                                         port=rethinkdb_config.port,
                                         db=rethinkdb_config.db)

    def get(self, provider: str, symbol: str, bin_size: str, start_date: int = None, end_date: int = None):
        query = self.r.table(f'trade_{provider.upper()}_{symbol.lower()}_{bin_size}')

        if start_date and end_date:
            start_date_dt = datetime.fromtimestamp(start_date / 1000).astimezone(pytz.UTC)
            end_date_dt = datetime.fromtimestamp(end_date / 1000).astimezone(pytz.UTC)
            query = (query
                     .filter(lambda t: t["timestamp"] >= start_date_dt)
                     .filter(lambda t: t["timestamp"] <= end_date_dt))

        return {
            'symbol': symbol.upper(),
            'bin_size': bin_size,
            'data': list(query.run(self.connection, time_format="raw"))}


class Optimization(Resource):

    def __init__(self, rethinkdb_config: RethinkDbConfig,
                 optimization_output: OptimizationOutputConfig,
                 walkforward_config: WalkForwardConfig):
        self.rethinkdb_config = rethinkdb_config
        self.optimization_output = optimization_output
        self.walkforward_config = walkforward_config

    def post(self):
        data = request.get_json()
        if not data:
            abort(400)
        try:

            engines = self._get_engines(kind=data['kind'])

            results = {}
            test_name = data['test_name']
            tid = hashlib.sha1(test_name.encode('utf-8')).hexdigest()
            results['tid'] = tid
            results['test_name'] = data['test_name']
            results['symbol'] = data['symbol']
            results['bin_size'] = data['bin_size']
            results['optimization_types'] = data['kind']
            results['status'] = 'submitted'

            for engine in engines:
                optimizer = Optimizer(rethinkdb_config=self.rethinkdb_config,
                                      optimization_output=self.optimization_output,
                                      walkforward_config=self.walkforward_config)
                optimizer.run(tid=tid,
                              test_name=test_name,
                              provider=DataSourceProviders[data['provider']],
                              symbol=data['symbol'],
                              bin_size=data['bin_size'],
                              strategy_name=data['strategy'],
                              strategy_params=data['strategy_params'],
                              cash=float(data['cash']),
                              commissions=float(data['commissions']),
                              start_date=datetime.fromtimestamp(data['start_date'] / 1000).astimezone(pytz.UTC),
                              end_date=datetime.fromtimestamp(data['end_date'] / 1000).astimezone(pytz.UTC),
                              engine=engine)

            return results
        except Exception as e:
            abort(500, error=str(e))

    @staticmethod
    def _get_engines(kind: str):
        if OptimizationType[kind] == OptimizationType.BACKTESTING:
            return [BacktestingEngine]
        elif OptimizationType[kind] == OptimizationType.WALKFORWARD:
            return [WalkForwardEngine]
        elif OptimizationType[kind] == OptimizationType.BOTH:
            return [BacktestingEngine, WalkForwardEngine]
        else:
            raise Exception(f'Unknown engine type {kind}')


class OptimizationProgress(Resource):

    def __init__(self, rethinkdb_config: RethinkDbConfig, optimization_output: OptimizationOutputConfig):
        self.rethinkdb_config = rethinkdb_config
        self.optimization_output = optimization_output
        self.r = RethinkDB()
        self.connection = self.r.connect(host=rethinkdb_config.host,
                                         port=rethinkdb_config.port,
                                         db=rethinkdb_config.db)

    def get(self, tid: str):
        res = self.r.table(self.optimization_output.progress).get(tid).run(self.connection)
        return res if res else {}


class OptimizationResult(Resource):

    def __init__(self, rethinkdb_config: RethinkDbConfig, optimization_output: OptimizationOutputConfig):
        self.rethinkdb_config = rethinkdb_config
        self.optimization_output = optimization_output
        self.r = RethinkDB()
        self.connection = self.r.connect(host=rethinkdb_config.host,
                                         port=rethinkdb_config.port,
                                         db=rethinkdb_config.db)

    def get(self, tid: str = None):
        try:
            if tid:
                res = (self.r.table(self.optimization_output.results)
                       .get(tid)
                       .without({'optimizations': {
                    OptimizationType.BACKTESTING.value: {'analyzers': {'pyfolio_sheets': True}},
                    OptimizationType.WALKFORWARD.value: {'analyzers': {'pyfolio_sheets': True}},
                }})
                       .run(self.connection))
                return res if res else {}
            else:
                res = (self.r.table(self.optimization_output.results)
                       .without({'optimizations': {
                    OptimizationType.BACKTESTING.value: {'analyzers': {'pyfolio_sheets': True}},
                    OptimizationType.WALKFORWARD.value: {'analyzers': {'pyfolio_sheets': True}}}})
                       .run(self.connection))
                return list(res) if res else []
        except ReqlOpFailedError as e:
            return []

    def delete(self, tid: str):
        return (self.r.table(self.optimization_output.results)
                .get(tid).delete().run(self.connection))


class OptimizationResultList(Resource):

    def __init__(self, rethinkdb_config: RethinkDbConfig, optimization_output: OptimizationOutputConfig):
        self.rethinkdb_config = rethinkdb_config
        self.optimization_output = optimization_output
        self.r = RethinkDB()
        self.connection = self.r.connect(host=rethinkdb_config.host,
                                         port=rethinkdb_config.port,
                                         db=rethinkdb_config.db)

    def get(self):
        try:
            res = (self.r.table(self.optimization_output.results)
                   .pluck('tid', 'test_name')
                   .run(self.connection))
            return list(res) if res else []
        except ReqlOpFailedError as e:
            return []


class OptimizationResultCorrelation(Resource):

    def __init__(self, rethinkdb_config: RethinkDbConfig,
                 optimization_output: OptimizationOutputConfig):
        self.rethinkdb_config = rethinkdb_config
        self.optimization_output = optimization_output
        self.r = RethinkDB()
        self.connection = self.r.connect(host=rethinkdb_config.host,
                                         port=rethinkdb_config.port,
                                         db=rethinkdb_config.db)

    def get(self, tid: str):
        try:
            if tid:
                res = self.r.table(self.optimization_output.results).get(tid).run(self.connection)

                records = []

                for t in res['optimizations']['BACKTESTING']:
                    dates = set()
                    for b in t['observers']['BuySell']['buy']:
                        records.append({'timestamp': b[0], 'test': -1, 'signal': 1})
                        dates.add(b[0])
                    for s in t['observers']['BuySell']['sell']:
                        records.append({'timestamp': s[0], 'test': -1, 'signal': -1})
                        dates.add(s[0])

                    start_dt = datetime.fromtimestamp(t['start_date'] / 1000)
                    end_dt = datetime.fromtimestamp(t['end_date'] / 1000)

                    while start_dt <= end_dt:
                        if start_dt not in dates:
                            records.append({'timestamp': start_dt.timestamp() * 1000, 'test': -1, 'signal': 0})
                        start_dt = start_dt + timedelta(days=1)

                for p, t in enumerate(res['optimizations']['WALKFORWARD'], start=1):
                    dates = set()
                    for b in t['observers']['BuySell']['buy']:
                        records.append({'timestamp': b[0], 'test': t['num_split'], 'signal': 1})
                        dates.add(b[0])
                    for s in t['observers']['BuySell']['sell']:
                        records.append({'timestamp': s[0], 'test': t['num_split'], 'signal': -1})
                        dates.add(s[0])

                    start_dt = datetime.fromtimestamp(t['start_date'] / 1000)
                    end_dt = datetime.fromtimestamp(t['end_date'] / 1000)

                    while start_dt <= end_dt:
                        if start_dt not in dates:
                            records.append({'timestamp': start_dt.timestamp() * 1000,
                                            'test': t['num_split'],
                                            'signal': 0})
                        start_dt = start_dt + timedelta(days=1)

                if len(records) == 0:
                    return {}

                records = sorted(records, key=lambda a: a['timestamp'])
                ws = pd.DataFrame(records)
                ws['timestamp'] = pd.to_datetime(ws['timestamp'], unit='ms')
                ws = ws.groupby('test').resample('1D', on='timestamp')['signal'].sum().reset_index()

                with StringIO() as buff:
                    ws[['timestamp', 'test', 'signal']].to_csv(buff, index=None)
                    return {'csv': buff.getvalue(),
                            'min_value': int(ws['signal'].min()),
                            'max_value': int(ws['signal'].max()),
                            'tests': sorted(map(int, ws['test'].unique())),
                            'min_ts': ws['timestamp'].min().timestamp(),
                            'max_ts': ws['timestamp'].max().timestamp(),
                            }

        except ReqlOpFailedError as e:
            return []


class OptimizationResultReport(Resource):

    def __init__(self, rethinkdb_config: RethinkDbConfig, optimization_output: OptimizationOutputConfig):
        self.rethinkdb_config = rethinkdb_config
        self.optimization_output = optimization_output
        self.r = RethinkDB()
        self.connection = self.r.connect(host=rethinkdb_config.host,
                                         port=rethinkdb_config.port,
                                         db=rethinkdb_config.db)

    def get(self, tid: str):
        try:
            res = (self.r.table(self.optimization_output.results)
                   .get(tid)
                   .run(self.connection))

            with BytesIO() as buff:
                workbook = xlsxwriter.Workbook(buff, {'default_date_format': 'mmm d yyyy hh:mm AM/PM',
                                                      'nan_inf_to_errors': True})
                self._get_summary_sheet(workbook=workbook, res=res)
                self._get_backtesting_sheet(workbook=workbook, res=res)
                self._get_walkforward_sheet(workbook=workbook, res=res)
                self._get_pyfolio_sheet(workbook=workbook, res=res)

                workbook.close()
                response = make_response(buff.getvalue())
                response.headers[
                    'content-type'] = "application/application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                response.headers[
                    'content-disposition'] = f"attachment; filename={res['test_name']}_{datetime.utcnow()}.xlsx"
                return response
        except ReqlOpFailedError as e:
            return []

    @staticmethod
    def _get_summary_sheet(workbook, res):
        bold = workbook.add_format({'bold': True})
        align = workbook.add_format({'align': 'right'})  # Widen the first column to make the text clearer.
        worksheet = workbook.add_worksheet(name='Summary')
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:B', 50)
        worksheet.write(0, 0, 'Test ID', bold)
        worksheet.write(0, 1, res['tid'], align)
        worksheet.write(1, 0, 'Test Name', bold)
        worksheet.write(1, 1, res['test_name'], align)
        worksheet.write(2, 0, 'Creation Time', bold)
        worksheet.write_datetime(2, 1, datetime.fromtimestamp(res['creation_time'] / 1000))
        worksheet.write(3, 0, 'Time Insterval', bold)
        worksheet.write(3, 1,
                        f'{datetime.fromtimestamp(res["start_date"] / 1000).strftime("%b %d %Y %H:%M %p")}'
                        f' - '
                        f'{datetime.fromtimestamp(res["end_date"] / 1000).strftime("%b %d %Y %H:%M %p")}'
                        , align)
        worksheet.write(4, 0, 'Provider', bold)
        worksheet.write(4, 1, res['provider'], align)
        worksheet.write(5, 0, 'Symbol', bold)
        worksheet.write(5, 1, res['symbol'], align)
        worksheet.write(6, 0, 'Timeframe', bold)
        worksheet.write(6, 1, res['timeframe'], align)
        worksheet.write(7, 0, 'Cash', bold)
        worksheet.write(7, 1, res['cash'], align)
        worksheet.write(8, 0, 'Commissions', bold)
        worksheet.write(8, 1, res['commissions'], align)

    def _get_backtesting_sheet(self, workbook, res):
        if OptimizationType.BACKTESTING.value not in res['optimizations']:
            return

        bold = workbook.add_format({'bold': True})
        align = workbook.add_format({'align': 'right', 'num_format': '0.000'})
        worksheet = workbook.add_worksheet(name='Backtesting')
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 20)
        worksheet.set_column('C:C', 15)
        worksheet.write(0, 0, 'Metric', bold)
        worksheet.write(0, 1, 'Value', bold)
        worksheet.write(0, 2, '$$', bold)

        perc_metrics = {
            'Annual return',
            'Annual volatility',
            'Cumulative returns',
            'Max drawdown',
            'Daily value at risk',
            'Daily turnover'
        }
        row = 1
        perc = workbook.add_format({'align': 'right', 'num_format': '0.00%'})
        money = workbook.add_format({'num_format': '$#,##0.00'})
        for m, v in res['optimizations'][OptimizationType.BACKTESTING.value][0]['analyzers']['PyFolio'].items():
            worksheet.write(row, 0, m, bold)
            if v:
                worksheet.write_number(row, 1, v, align if m not in perc_metrics else perc)
                if m in perc_metrics:
                    worksheet.write_number(row, 2, v * res['cash'], money)

            row += 1

    def _get_walkforward_sheet(self, workbook, res):
        if OptimizationType.WALKFORWARD.value not in res['optimizations']:
            return

        bold = workbook.add_format({'bold': True})
        align = workbook.add_format({'align': 'right', 'num_format': '0.000'})
        worksheet = workbook.add_worksheet(name='Walkforward')
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 20)
        worksheet.set_column('C:Z', 15)
        worksheet.write(0, 0, 'Metric', bold)
        worksheet.write(0, 1, 'Value', bold)
        worksheet.write(0, 2, 'Std. Dev', bold)
        worksheet.write(0, 3, '$$', bold)

        perc_metrics = {
            'Annual return',
            'Annual volatility',
            'Cumulative returns',
            'Max drawdown',
            'Daily value at risk',
            'Daily turnover'
        }

        perc = workbook.add_format({'align': 'right', 'num_format': '0.00%'})
        money = workbook.add_format({'num_format': '$#,##0.00'})
        stats = defaultdict(list)

        for col, test in enumerate(res['optimizations'][OptimizationType.WALKFORWARD.value], start=4):
            worksheet.write(0, col, f'Out-of-sample-{col - 4}', bold)
            for row, (m, v) in enumerate(test['analyzers']['PyFolio'].items(), start=1):
                worksheet.write(row, 0, m, bold)
                if v:
                    worksheet.write_number(row, col, v, align if m not in perc_metrics else perc)
                    stats[m].append(v)

        for row, (stat, values) in enumerate(stats.items(), start=1):
            mean = np.nanmean(values)
            worksheet.write(row, 1, mean, align if stat not in perc_metrics else perc)
            worksheet.write(row, 2, np.nanstd(values), align if stat not in perc_metrics else perc)
            if stat in perc_metrics:
                worksheet.write_number(row, 3, mean * res['cash'], money)

    def _get_pyfolio_sheet(self, workbook, res):
        if OptimizationType.BACKTESTING.value not in res['optimizations']:
            return

        for name, img in res['optimizations'][OptimizationType.BACKTESTING.value][0]['analyzers'][
            'pyfolio_sheets'].items():
            worksheet = workbook.add_worksheet(name=f'{OptimizationType.BACKTESTING.value.lower()}_{name}')
            buff = BytesIO(base64.b64decode(img))
            worksheet.insert_image('B2', name, {'image_data': buff})

        if OptimizationType.WALKFORWARD.value not in res['optimizations']:
            return

        for test in res['optimizations'][OptimizationType.WALKFORWARD.value]:
            for name, img in test['analyzers']['pyfolio_sheets'].items():
                worksheet = workbook.add_worksheet(name=f'wf_{test["num_split"]}_{name}')
                buff = BytesIO(base64.b64decode(img))
                worksheet.insert_image('B2', name, {'image_data': buff})


if __name__ == '__main__':
    CONFIG_FILE = os.environ.get('CONFIG_FILE', 'maestro-dev.yaml')

    path = ROOT_DIR.joinpath(CONFIG_FILE)

    config_reader = ConfigReader().read(path=path)
    rest_api_config = config_reader.get_restapi_config()
    rethinkdb_config = config_reader.get_rethinkdb_config()

    api.add_resource(StrategyList, '/strategy/available')
    api.add_resource(StrategyParams, '/strategy/<string:strategy>/params')
    api.add_resource(Optimization, '/optimization/new/',
                     resource_class_kwargs={'rethinkdb_config': rethinkdb_config,
                                            'optimization_output': config_reader.get_optimization_output_config(),
                                            'walkforward_config': config_reader.get_walkforward_config()
                                            })

    api.add_resource(DataSourceProvider, '/datasource/available')

    api.add_resource(DataSourceProviderSymbol, '/datasource/<string:provider>/symbols',
                     resource_class_kwargs={'rethinkdb_config': rethinkdb_config})

    api.add_resource(DataSource,
                     '/datasource/<string:provider>/<string:symbol>/<string:bin_size>/',
                     '/datasource/<string:provider>/<string:symbol>/<string:bin_size>/<int:start_date>/<int:end_date>',
                     resource_class_kwargs={'rethinkdb_config': rethinkdb_config})

    api.add_resource(OptimizationProgress, '/optimization/progress/<string:tid>/',
                     resource_class_kwargs={'rethinkdb_config': rethinkdb_config,
                                            'optimization_output': config_reader.get_optimization_output_config()})

    api.add_resource(OptimizationResult, '/optimization/results', '/optimization/results/<string:tid>',
                     resource_class_kwargs={'rethinkdb_config': rethinkdb_config,
                                            'optimization_output': config_reader.get_optimization_output_config()})

    api.add_resource(OptimizationResultCorrelation, '/optimization/results/<string:tid>/correlation',
                     resource_class_kwargs={'rethinkdb_config': rethinkdb_config,
                                            'optimization_output': config_reader.get_optimization_output_config()})

    api.add_resource(OptimizationResultList, '/optimization/results/available',
                     resource_class_kwargs={'rethinkdb_config': rethinkdb_config,
                                            'optimization_output': config_reader.get_optimization_output_config()}
                     )

    api.add_resource(OptimizationResultReport,
                     '/optimization/results/<string:tid>/report',
                     resource_class_kwargs={'rethinkdb_config': rethinkdb_config,
                                            'optimization_output': config_reader.get_optimization_output_config()}
                     )

    app.run(host=rest_api_config.host, port=rest_api_config.port, debug=rest_api_config.debug)
