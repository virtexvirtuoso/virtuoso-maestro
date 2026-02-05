"""
Migration script for RethinkDB to QuestDB data transfer with validation.

Features:
- Dry-run mode for validation only
- SHA256 checksum validation
- Float precision verification (< 1e-10 tolerance)
- 100% row count matching
- ILP-based ingestion for performance
- Progress logging with summary report

Usage:
    python backend/scripts/migrate_rethinkdb_to_questdb.py --dry-run
    python backend/scripts/migrate_rethinkdb_to_questdb.py --table trade_binance_ethbtc_1d
    python backend/scripts/migrate_rethinkdb_to_questdb.py
"""

from __future__ import annotations
import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from rethinkdb import RethinkDB

from storage.questdb_storage_layer import QuestDBStorageLayer
from logger.logger_builder import LoggerBuilder


@dataclass
class ValidationResult:
    """Result of validating a single table migration."""
    table_name: str
    rethink_row_count: int
    questdb_row_count: int
    row_count_match: bool
    rethink_checksum: str
    questdb_checksum: str
    checksum_match: bool
    float_precision_issues: List[str] = field(default_factory=list)
    passed: bool = False
    error: Optional[str] = None

    def __post_init__(self):
        self.passed = (
            self.row_count_match
            and self.checksum_match
            and len(self.float_precision_issues) == 0
            and self.error is None
        )


@dataclass
class MigrationSummary:
    """Summary of full migration run."""
    total_tables: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    dry_run: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class MigrationValidator:
    """Validates data integrity between RethinkDB and QuestDB."""

    FLOAT_TOLERANCE = 1e-10
    OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def calculate_checksum(self, df: pd.DataFrame) -> str:
        """
        Calculate SHA256 checksum of timestamp+close+volume columns.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Hex digest of SHA256 hash
        """
        if df.empty:
            return hashlib.sha256(b'').hexdigest()

        # Sort by timestamp for deterministic ordering
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)

        # Extract checksum columns: timestamp, close, volume
        checksum_cols = ['timestamp', 'close', 'volume']
        available_cols = [c for c in checksum_cols if c in df_sorted.columns]

        if not available_cols:
            return hashlib.sha256(b'').hexdigest()

        # Convert to string representation for hashing
        # Normalize timestamp to epoch milliseconds
        checksum_data = []
        for _, row in df_sorted.iterrows():
            row_data = []
            for col in available_cols:
                val = row[col]
                if col == 'timestamp':
                    # Normalize to epoch ms
                    if isinstance(val, datetime):
                        val = int(val.timestamp() * 1000)
                    elif isinstance(val, pd.Timestamp):
                        val = int(val.timestamp() * 1000)
                    elif isinstance(val, (int, float)):
                        # Assume already ms or convert
                        val = int(val)
                    else:
                        val = str(val)
                else:
                    # Format floats consistently to 10 decimal places
                    val = f'{float(val):.10f}'
                row_data.append(str(val))
            checksum_data.append('|'.join(row_data))

        data_str = '\n'.join(checksum_data)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def check_float_precision(
        self,
        df_rethink: pd.DataFrame,
        df_questdb: pd.DataFrame
    ) -> List[str]:
        """
        Check float precision differences between DataFrames.

        Args:
            df_rethink: RethinkDB data
            df_questdb: QuestDB data

        Returns:
            List of precision issue descriptions
        """
        issues = []

        if df_rethink.empty or df_questdb.empty:
            return issues

        # Sort both by timestamp for row-by-row comparison
        df_r = df_rethink.sort_values('timestamp').reset_index(drop=True)
        df_q = df_questdb.sort_values('timestamp').reset_index(drop=True)

        if len(df_r) != len(df_q):
            issues.append(f'Row count mismatch: {len(df_r)} vs {len(df_q)}')
            return issues

        for col in self.OHLCV_COLUMNS:
            if col not in df_r.columns or col not in df_q.columns:
                continue

            for idx in range(len(df_r)):
                val_r = float(df_r.loc[idx, col])
                val_q = float(df_q.loc[idx, col])
                diff = abs(val_r - val_q)

                if diff > self.FLOAT_TOLERANCE:
                    issues.append(
                        f'Row {idx}, column {col}: diff={diff:.2e} '
                        f'(rethink={val_r}, questdb={val_q})'
                    )
                    # Limit reported issues
                    if len(issues) >= 10:
                        issues.append('... (truncated, more issues exist)')
                        return issues

        return issues

    def validate_migration(
        self,
        table_name: str,
        df_rethink: pd.DataFrame,
        df_questdb: pd.DataFrame
    ) -> ValidationResult:
        """
        Validate migration of a single table.

        Args:
            table_name: Name of the table
            df_rethink: Data from RethinkDB
            df_questdb: Data from QuestDB

        Returns:
            ValidationResult with all checks
        """
        rethink_count = len(df_rethink)
        questdb_count = len(df_questdb)
        row_count_match = rethink_count == questdb_count

        rethink_checksum = self.calculate_checksum(df_rethink)
        questdb_checksum = self.calculate_checksum(df_questdb)
        checksum_match = rethink_checksum == questdb_checksum

        precision_issues = self.check_float_precision(df_rethink, df_questdb)

        return ValidationResult(
            table_name=table_name,
            rethink_row_count=rethink_count,
            questdb_row_count=questdb_count,
            row_count_match=row_count_match,
            rethink_checksum=rethink_checksum,
            questdb_checksum=questdb_checksum,
            checksum_match=checksum_match,
            float_precision_issues=precision_issues
        )


class RethinkDBToQuestDBMigrator:
    """Handles migration of trade data from RethinkDB to QuestDB."""

    METADATA_TABLE = 'trade_metadata'

    def __init__(
        self,
        rethink_host: str = '127.0.0.1',
        rethink_port: int = 28015,
        rethink_db: str = 'filos-dev',
        questdb_host: str = 'localhost',
        questdb_pg_port: int = 8812,
        questdb_ilp_port: int = 9009,
        logger: logging.Logger = None
    ):
        self.rethink_host = rethink_host
        self.rethink_port = rethink_port
        self.rethink_db = rethink_db
        self.logger = logger or logging.getLogger(__name__)
        self.validator = MigrationValidator(self.logger)

        # Initialize QuestDB connection
        self.questdb = QuestDBStorageLayer(
            host=questdb_host,
            pg_port=questdb_pg_port,
            ilp_port=questdb_ilp_port,
            logger=self.logger
        )

        # RethinkDB connection
        self._r = RethinkDB()
        self._rethink_conn = None

    def _get_rethink_connection(self):
        """Get or create RethinkDB connection."""
        if self._rethink_conn is None or not self._rethink_conn.is_open():
            self._rethink_conn = self._r.connect(
                host=self.rethink_host,
                port=self.rethink_port,
                db=self.rethink_db
            )
        return self._rethink_conn

    def get_trade_tables(self) -> List[str]:
        """Get all trade_ tables from RethinkDB (excluding metadata)."""
        conn = self._get_rethink_connection()
        tables = list(self._r.table_list().run(conn))
        trade_tables = [
            t for t in tables
            if t.startswith('trade_') and t != self.METADATA_TABLE
        ]
        return sorted(trade_tables)

    def read_rethink_table(self, table_name: str) -> pd.DataFrame:
        """
        Read all data from a RethinkDB table.

        Args:
            table_name: Name of the table to read

        Returns:
            DataFrame with OHLCV data
        """
        conn = self._get_rethink_connection()
        cursor = self._r.table(table_name).run(conn)
        records = list(cursor)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Normalize column names (RethinkDB might have different casing)
        df.columns = [c.lower() for c in df.columns]

        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            self.logger.warning(f'{table_name}: No timestamp column found')
            return pd.DataFrame()

        return df

    def read_questdb_table(self, table_name: str) -> pd.DataFrame:
        """
        Read all data from a QuestDB table.

        Args:
            table_name: Name of the table to read

        Returns:
            DataFrame with OHLCV data
        """
        return self.questdb.query_ohlcv(table_name)

    def migrate_table(
        self,
        table_name: str,
        dry_run: bool = False
    ) -> ValidationResult:
        """
        Migrate a single table from RethinkDB to QuestDB.

        Args:
            table_name: Name of the table to migrate
            dry_run: If True, only validate without writing

        Returns:
            ValidationResult with migration status
        """
        self.logger.info(f'[{table_name}] Starting migration (dry_run={dry_run})')

        try:
            # Read from RethinkDB
            df_rethink = self.read_rethink_table(table_name)
            self.logger.info(f'[{table_name}] Read {len(df_rethink)} rows from RethinkDB')

            if df_rethink.empty:
                self.logger.warning(f'[{table_name}] No data to migrate')
                return ValidationResult(
                    table_name=table_name,
                    rethink_row_count=0,
                    questdb_row_count=0,
                    row_count_match=True,
                    rethink_checksum='',
                    questdb_checksum='',
                    checksum_match=True,
                    error='No data in source table'
                )

            if not dry_run:
                # Create table in QuestDB (drop if exists for clean migration)
                if self.questdb.table_exists(table_name):
                    self.logger.info(f'[{table_name}] Dropping existing QuestDB table')
                    self.questdb.drop_table(table_name)

                # Create fresh table
                self.questdb.create_trade_table(table_name)

                # Ingest via ILP
                rows_ingested = self.questdb.ingest_dataframe(table_name, df_rethink)
                self.logger.info(f'[{table_name}] Ingested {rows_ingested} rows to QuestDB')

                # Small delay to ensure QuestDB commits
                import time
                time.sleep(0.5)

            # Read back from QuestDB for validation
            if dry_run:
                # In dry-run, check if table exists and read it
                if self.questdb.table_exists(table_name):
                    df_questdb = self.read_questdb_table(table_name)
                else:
                    df_questdb = pd.DataFrame()
                    self.logger.info(f'[{table_name}] QuestDB table does not exist (dry-run)')
            else:
                df_questdb = self.read_questdb_table(table_name)
                self.logger.info(f'[{table_name}] Read {len(df_questdb)} rows from QuestDB')

            # Validate
            result = self.validator.validate_migration(
                table_name=table_name,
                df_rethink=df_rethink,
                df_questdb=df_questdb
            )

            if result.passed:
                self.logger.info(f'[{table_name}] Validation PASSED')
            else:
                self.logger.error(f'[{table_name}] Validation FAILED')
                if not result.row_count_match:
                    self.logger.error(
                        f'  Row count: {result.rethink_row_count} vs {result.questdb_row_count}'
                    )
                if not result.checksum_match:
                    self.logger.error(f'  Checksum mismatch')
                for issue in result.float_precision_issues[:5]:
                    self.logger.error(f'  {issue}')

            return result

        except Exception as e:
            self.logger.exception(f'[{table_name}] Migration failed with error')
            return ValidationResult(
                table_name=table_name,
                rethink_row_count=0,
                questdb_row_count=0,
                row_count_match=False,
                rethink_checksum='',
                questdb_checksum='',
                checksum_match=False,
                error=str(e)
            )

    def migrate_all(
        self,
        table_filter: Optional[str] = None,
        dry_run: bool = False
    ) -> MigrationSummary:
        """
        Migrate all tables (or a single table) from RethinkDB to QuestDB.

        Args:
            table_filter: If provided, only migrate this table
            dry_run: If True, only validate without writing

        Returns:
            MigrationSummary with results for all tables
        """
        summary = MigrationSummary(dry_run=dry_run)
        summary.start_time = datetime.now()

        # Get tables to migrate
        if table_filter:
            tables = [table_filter] if table_filter in self.get_trade_tables() else []
            if not tables:
                self.logger.error(f'Table {table_filter} not found in RethinkDB')
                tables = [table_filter]  # Try anyway for error reporting
        else:
            tables = self.get_trade_tables()

        summary.total_tables = len(tables)
        self.logger.info(f'Found {summary.total_tables} tables to migrate')

        # Process each table
        for i, table_name in enumerate(tables, 1):
            self.logger.info(f'Processing table {i}/{summary.total_tables}: {table_name}')

            result = self.migrate_table(table_name, dry_run=dry_run)
            summary.results.append(result)

            if result.passed:
                summary.successful += 1
            elif result.error and 'No data' in result.error:
                summary.skipped += 1
            else:
                summary.failed += 1

        summary.end_time = datetime.now()
        return summary

    def close(self):
        """Close all connections."""
        if self._rethink_conn and self._rethink_conn.is_open():
            self._rethink_conn.close()
        self.questdb.close()


def print_summary(summary: MigrationSummary, logger: logging.Logger):
    """Print migration summary report."""
    logger.info('=' * 60)
    logger.info('MIGRATION SUMMARY')
    logger.info('=' * 60)
    logger.info(f'Mode: {"DRY RUN" if summary.dry_run else "LIVE MIGRATION"}')
    logger.info(f'Duration: {summary.duration_seconds:.2f} seconds')
    logger.info(f'Total tables: {summary.total_tables}')
    logger.info(f'Successful: {summary.successful}')
    logger.info(f'Failed: {summary.failed}')
    logger.info(f'Skipped (empty): {summary.skipped}')
    logger.info('-' * 60)

    if summary.failed > 0:
        logger.info('FAILED TABLES:')
        for result in summary.results:
            if not result.passed and result.error is None:
                logger.info(f'  - {result.table_name}')
                if not result.row_count_match:
                    logger.info(
                        f'    Row count: {result.rethink_row_count} vs {result.questdb_row_count}'
                    )
                if not result.checksum_match:
                    logger.info('    Checksum mismatch')
                for issue in result.float_precision_issues[:3]:
                    logger.info(f'    {issue}')

    if summary.results and all(r.passed or r.error for r in summary.results):
        logger.info('All validations PASSED!')
    logger.info('=' * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Migrate trade data from RethinkDB to QuestDB with validation'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate only, do not write to QuestDB'
    )
    parser.add_argument(
        '--table',
        type=str,
        default=None,
        help='Migrate only this specific table'
    )
    parser.add_argument(
        '--rethink-host',
        type=str,
        default='127.0.0.1',
        help='RethinkDB host (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--rethink-port',
        type=int,
        default=28015,
        help='RethinkDB port (default: 28015)'
    )
    parser.add_argument(
        '--rethink-db',
        type=str,
        default='filos-dev',
        help='RethinkDB database name (default: filos-dev)'
    )
    parser.add_argument(
        '--questdb-host',
        type=str,
        default='localhost',
        help='QuestDB host (default: localhost)'
    )
    parser.add_argument(
        '--questdb-pg-port',
        type=int,
        default=8812,
        help='QuestDB PostgreSQL port (default: 8812)'
    )
    parser.add_argument(
        '--questdb-ilp-port',
        type=int,
        default=9009,
        help='QuestDB ILP port (default: 9009)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logger = LoggerBuilder('migration').with_level(logging.INFO).build()

    logger.info('RethinkDB to QuestDB Migration Script')
    logger.info(f'Dry run: {args.dry_run}')
    logger.info(f'Table filter: {args.table or "all tables"}')

    try:
        migrator = RethinkDBToQuestDBMigrator(
            rethink_host=args.rethink_host,
            rethink_port=args.rethink_port,
            rethink_db=args.rethink_db,
            questdb_host=args.questdb_host,
            questdb_pg_port=args.questdb_pg_port,
            questdb_ilp_port=args.questdb_ilp_port,
            logger=logger
        )

        summary = migrator.migrate_all(
            table_filter=args.table,
            dry_run=args.dry_run
        )

        print_summary(summary, logger)

        # Exit with error code if any failures
        if summary.failed > 0:
            sys.exit(1)

    except Exception as e:
        logger.exception('Migration failed')
        sys.exit(1)

    finally:
        if 'migrator' in locals():
            migrator.close()


if __name__ == '__main__':
    main()
