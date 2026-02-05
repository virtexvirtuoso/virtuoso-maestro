from datafeed.rethinkdb_datafeed_builder import RethinkDBDataFeedBuilder
from engine.optimization_engine import OptimizationEngine
from engine.optimizer import OptimizationType
import traceback


class BacktestingEngine(OptimizationEngine):

    def run(self):
        try:
            cerebro = self.get_cerebro()

            # Add a strategy
            self.add_strategy(cerebro=cerebro)

            data = (RethinkDBDataFeedBuilder(host=self.rethinkdb_config.host,
                                             port=self.rethinkdb_config.port,
                                             db=self.rethinkdb_config.db)
                    .build(provider=self.provider,
                           symbol=self.symbol,
                           bin_size=self.bin_size,
                           start_date=self.start_date,
                           end_date=self.end_date))

            # Add the Data Feed to Cerebro
            cerebro.adddata(data)
            self.total_bars = len(data.p.dataname) + 1
            self.run_and_save(cerebro=cerebro, num_split=None)
        except Exception as e:
            self.logger.error(f'Error during {self.__class__.__name__}.\n {traceback.format_exc()}')
        finally:
            self.cur_bar = self.total_bars - 1
            self.update_progress()

    def update_progress(self):
        self.cur_bar += 1
        self.save_progress_to_db(cur=self.cur_bar, total=self.total_bars)

    @property
    def kind(self):
        return OptimizationType.BACKTESTING
