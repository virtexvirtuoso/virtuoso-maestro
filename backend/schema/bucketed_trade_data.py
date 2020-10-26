from dataclasses import dataclass


@dataclass
class BucketedTradeData:
    symbol: str
    bin_size: str
    start: int

    def __repr__(self) -> str:
        return f'{self.symbol} - {self.bin_size} [start={self.start}]'
