"""Maestro DuckDB data loader for backtests."""

import duckdb
import pandas as pd
import os


class MaestroDB:
    def __init__(self, db_path="~/Desktop/maestro/data/maestro.duckdb"):
        self.db_path = os.path.expanduser(db_path)
        self.con = duckdb.connect(self.db_path, read_only=True)

    def _query(self, sql, params=None):
        return self.con.execute(sql, params or []).fetchdf()

    def _date_filter(self, start=None, end=None):
        clauses, params = [], []
        if start:
            clauses.append("date >= ?")
            params.append(start)
        if end:
            clauses.append("date <= ?")
            params.append(end)
        return (" AND " + " AND ".join(clauses) if clauses else ""), params

    def spot(self, symbol, start=None, end=None) -> pd.DataFrame:
        filt, params = self._date_filter(start, end)
        return self._query(
            f"SELECT * FROM spot_daily WHERE symbol = ?{filt} ORDER BY date",
            [symbol.upper()] + params
        )

    def perps(self, symbol, start=None, end=None) -> pd.DataFrame:
        filt, params = self._date_filter(start, end)
        try:
            return self._query(
                f"SELECT * FROM perps_daily WHERE symbol = ?{filt} ORDER BY date",
                [symbol.upper()] + params
            )
        except Exception:
            return pd.DataFrame()

    def funding(self, symbol, source="coinglass", start=None, end=None) -> pd.DataFrame:
        table = {"coinglass": "cg_funding_rate", "coinalyze": "coinalyze_funding",
                 "binance": "funding_rates_binance"}.get(source, "cg_funding_rate")
        filt, params = self._date_filter(start, end)
        try:
            return self._query(
                f"SELECT * FROM {table} WHERE symbol = ?{filt} ORDER BY date",
                [symbol.upper()] + params
            )
        except Exception:
            return pd.DataFrame()

    def lsr(self, symbol, type="global", start=None, end=None) -> pd.DataFrame:
        table = {"global": "cg_lsr_global", "top_account": "cg_lsr_top_account",
                 "top_position": "cg_lsr_top_position"}.get(type, "cg_lsr_global")
        filt, params = self._date_filter(start, end)
        return self._query(
            f"SELECT * FROM {table} WHERE symbol = ?{filt} ORDER BY date",
            [symbol.upper()] + params
        )

    def liquidations(self, symbol, source="coinglass", start=None, end=None) -> pd.DataFrame:
        table = "cg_liquidations" if source == "coinglass" else "coinalyze_liquidations"
        filt, params = self._date_filter(start, end)
        return self._query(
            f"SELECT * FROM {table} WHERE symbol = ?{filt} ORDER BY date",
            [symbol.upper()] + params
        )

    def oi(self, symbol, source="coinglass", start=None, end=None) -> pd.DataFrame:
        table = "cg_oi_ohlc" if source == "coinglass" else "coinalyze_oi"
        filt, params = self._date_filter(start, end)
        try:
            return self._query(
                f"SELECT * FROM {table} WHERE symbol = ?{filt} ORDER BY date",
                [symbol.upper()] + params
            )
        except Exception:
            return pd.DataFrame()

    def taker_volume(self, symbol, start=None, end=None) -> pd.DataFrame:
        filt, params = self._date_filter(start, end)
        return self._query(
            f"SELECT * FROM cg_taker_volume WHERE symbol = ?{filt} ORDER BY date",
            [symbol.upper()] + params
        )

    def fear_greed(self, start=None, end=None) -> pd.DataFrame:
        filt, params = self._date_filter(start, end)
        try:
            return self._query(f"SELECT * FROM fear_greed WHERE 1=1{filt} ORDER BY date", params)
        except Exception:
            return pd.DataFrame()

    def whale_trades(self, symbol=None, start=None, end=None) -> pd.DataFrame:
        filt, params = self._date_filter(start, end)
        sym_clause = ""
        if symbol:
            sym_clause = " AND symbol = ?"
            params = [symbol.upper()] + params
        return self._query(
            f"SELECT * FROM whale_trades WHERE 1=1{sym_clause}{filt} ORDER BY date",
            params
        )

    def query(self, sql) -> pd.DataFrame:
        return self._query(sql)

    def summary(self) -> str:
        tables = self.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' AND table_name != '_metadata' ORDER BY table_name"
        ).fetchall()
        lines = [f"{'Table':<30} {'Rows':>10} {'Date Range':<35}"]
        lines.append("=" * 75)
        for (t,) in tables:
            rc = self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            try:
                cols = [r[0] for r in self.con.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name='{t}'").fetchall()]
                if "date" in cols:
                    dr = self.con.execute(f"SELECT MIN(date), MAX(date) FROM {t}").fetchone()
                    date_range = f"{dr[0]} → {dr[1]}" if dr[0] else "N/A"
                else:
                    date_range = "no date col"
            except Exception as e:
                date_range = f"err: {e}"
            lines.append(f"{t:<30} {rc:>10,} {date_range:<35}")
        result = "\n".join(lines)
        print(result)
        return result

    def close(self):
        self.con.close()

    def __del__(self):
        try:
            self.con.close()
        except Exception:
            pass
