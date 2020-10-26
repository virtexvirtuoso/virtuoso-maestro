import backtrader.analyzers as btanalyzers
import pandas as pd
import traceback
from datafeed.rethinkdb_datafeed_builder import RethinkDBDataFeedBuilder
from engine.optimization_engine import OptimizationEngine
from engine.optimizer import OptimizationType
from utils.time_series_split_rolling import TimeSeriesSplitRolling


class WalkForwardEngine(OptimizationEngine):

    def run(self):
        data_builder = RethinkDBDataFeedBuilder(host=self.rethinkdb_config.host,
                                                port=self.rethinkdb_config.port,
                                                db=self.rethinkdb_config.db)

        df = data_builder.read_data_frame(provider=self.provider,
                                          symbol=self.symbol,
                                          bin_size=self.bin_size,
                                          start_date=self.start_date,
                                          end_date=self.end_date)

        num_split = self.walkforward_config.num_splits
        tsrcv = TimeSeriesSplitRolling(num_split)
        split = tsrcv.split(df, fixed_length=True, train_splits=2)
        self.total_bars = num_split

        # Be prepared: this will take a while
        for pos, (train, test) in enumerate(split):
            try:
                # TRAINING
                trainer = self.get_cerebro()
                self.add_opt_strategy(cerebro=trainer, strategy=self.strategy, data_size=len(df.iloc[train]))

                training_data = data_builder.to_datafeed(provider=self.provider,
                                                         symbol=self.symbol,
                                                         bin_size=self.bin_size,
                                                         df=df.iloc[train])
                # to the object that
                # corresponds to training
                trainer.adddata(training_data)
                trainer.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_ratio', timeframe=btanalyzers.TimeFrame.Days)
                trainer.addanalyzer(btanalyzers.VWR, _name='vwr')

                res = trainer.run()

                if len(res):
                    data = []

                    for r in res:
                        d = {
                            'sharpe_ratio': r[0].analyzers.sharpe_ratio.get_analysis()['sharperatio'],
                            'vwr': r[0].analyzers.vwr.get_analysis()['vwr']
                        }
                        for p in self.strategy.get_params().keys():
                            d[p] = getattr(r[0].params, p)

                        data.append(d)

                    opt_res = pd.DataFrame(data=data)

                    # Optimal Parameters
                    opt_res = opt_res.sort_values(by=['sharpe_ratio', 'vwr'],
                                                  ascending=[False, False]).drop(columns=['sharpe_ratio', 'vwr']).iloc[
                        0].to_dict()
                else:
                    opt_res = self.strategy_params

                # TESTING
                self.cur_bar += 1
                tester = self.get_cerebro()
                self.add_strategy(cerebro=tester,
                                  opt_params=opt_res)

                test_data = data_builder.to_datafeed(provider=self.provider,
                                                     symbol=self.symbol,
                                                     bin_size=self.bin_size,
                                                     df=df.iloc[test])
                # to the object that
                # corresponds to training
                tester.adddata(test_data)
                self.run_and_save(cerebro=tester, num_split=pos)
            except Exception as e:
                self.logger.error(f'Error during {self.__class__.__name__} {num_split}.\n {traceback.format_exc()}')
                self.cur_bar += 1
                self.update_progress()

        self.cur_bar = self.total_bars
        self.update_progress()

    @property
    def kind(self):
        return OptimizationType.WALKFORWARD

    def update_progress(self):
        self.save_progress_to_db(cur=self.cur_bar, total=self.total_bars)
