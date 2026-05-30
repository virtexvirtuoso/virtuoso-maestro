"""
Optuna Dashboard Storage - SQLite storage with dashboard support

This module provides persistent SQLite storage for Optuna studies, enabling
visualization through the Optuna Dashboard web UI.

Features:
- Persistent SQLite storage at configurable path
- Study naming convention: maestro_fold_{fold_idx}
- Warm-starting with load_if_exists=True
- Dashboard-compatible storage URL generation

Usage:
    # Create a study with dashboard support
    storage = OptunaDashboardStorage()
    study = storage.create_study_with_dashboard(fold_idx=0)

    # Run optimization
    study.optimize(objective, n_trials=100)

    # Launch dashboard (CLI):
    # optuna-dashboard sqlite:///data/optuna_studies.db --port 8050
"""

import logging
import os
from pathlib import Path
from typing import Optional

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler

# Default storage path relative to backend directory
DEFAULT_STORAGE_DIR = "data"
DEFAULT_STORAGE_FILENAME = "optuna_studies.db"


def get_default_storage_path() -> str:
    """
    Get the default storage path for Optuna studies.

    Returns path relative to the backend directory, creating
    the data directory if it doesn't exist.
    """
    # Determine backend root
    backend_root = Path(__file__).parent.parent
    storage_dir = backend_root / DEFAULT_STORAGE_DIR

    # Create directory if needed
    storage_dir.mkdir(parents=True, exist_ok=True)

    return str(storage_dir / DEFAULT_STORAGE_FILENAME)


def get_storage_url(db_path: str | None = None) -> str:
    """
    Get SQLite storage URL for Optuna.

    Args:
        db_path: Path to SQLite database file. Uses default if not provided.

    Returns:
        SQLite URL in format: sqlite:///path/to/optuna_studies.db
    """
    path = db_path or get_default_storage_path()
    return f"sqlite:///{path}"


def _enable_wal_mode(db_path: str) -> None:
    """Enable WAL journal mode on SQLite DB for safe concurrent writes (Phase 1)."""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
    except Exception:
        pass  # Best-effort — DB may not exist yet


class OptunaDashboardStorage:
    """
    Manages Optuna study storage with dashboard support.

    Provides a centralized way to create and manage Optuna studies
    that persist to SQLite for dashboard visualization.

    Example:
        storage = OptunaDashboardStorage()

        # Create study for fold 0
        study = storage.create_study_with_dashboard(fold_idx=0)

        # Run optimization
        study.optimize(objective, n_trials=100)

        # Get storage URL for dashboard
        print(storage.storage_url)  # sqlite:///data/optuna_studies.db
    """

    # Study name prefix for Maestro
    STUDY_PREFIX = "maestro_fold_"

    def __init__(
        self,
        db_path: str | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize storage manager.

        Args:
            db_path: Path to SQLite database. Uses default if not provided.
            logger: Logger instance. Creates default if not provided.
        """
        self.db_path = db_path or get_default_storage_path()
        self.storage_url = get_storage_url(self.db_path)
        self.logger = logger or logging.getLogger(__name__)

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Optuna storage initialized: {self.storage_url}")

    def create_study_with_dashboard(
        self,
        fold_idx: int,
        direction: str = "maximize",
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        load_if_exists: bool = True,
        study_name_override: str | None = None,
    ) -> optuna.Study:
        """
        Create an Optuna study with dashboard-compatible SQLite storage.

        Args:
            fold_idx: Fold index for study naming (maestro_fold_{fold_idx})
            direction: Optimization direction ("maximize" or "minimize")
            sampler: Optuna sampler. Defaults to TPESampler with seed.
            pruner: Optuna pruner. Defaults to MedianPruner.
            load_if_exists: If True, load existing study for warm-starting.
            study_name_override: Override the default study name.

        Returns:
            optuna.Study instance with SQLite storage
        """
        # Generate study name
        study_name = study_name_override or f"{self.STUDY_PREFIX}{fold_idx}"

        # Default sampler with reproducible seed
        if sampler is None:
            sampler = TPESampler(seed=42 + fold_idx)

        # Default pruner
        if pruner is None:
            pruner = MedianPruner(n_startup_trials=10)

        self.logger.info(
            f"Creating Optuna study: {study_name} "
            f"(storage: {self.storage_url}, load_if_exists: {load_if_exists})"
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=load_if_exists,
        )

        return study

    def get_study(
        self,
        fold_idx: int,
        study_name_override: str | None = None,
    ) -> optuna.Study | None:
        """
        Load an existing study by fold index.

        Args:
            fold_idx: Fold index
            study_name_override: Override the default study name.

        Returns:
            optuna.Study if exists, None otherwise
        """
        study_name = study_name_override or f"{self.STUDY_PREFIX}{fold_idx}"

        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage_url,
            )
            return study
        except KeyError:
            self.logger.debug(f"Study not found: {study_name}")
            return None

    def list_studies(self) -> list:
        """
        List all studies in storage.

        Returns:
            List of study summaries
        """
        try:
            summaries = optuna.get_all_study_summaries(storage=self.storage_url)
            return summaries
        except Exception as e:
            self.logger.warning(f"Failed to list studies: {e}")
            return []

    def delete_study(
        self,
        fold_idx: int,
        study_name_override: str | None = None,
    ) -> bool:
        """
        Delete a study by fold index.

        Args:
            fold_idx: Fold index
            study_name_override: Override the default study name.

        Returns:
            True if deleted, False if not found
        """
        study_name = study_name_override or f"{self.STUDY_PREFIX}{fold_idx}"

        try:
            optuna.delete_study(
                study_name=study_name,
                storage=self.storage_url,
            )
            self.logger.info(f"Deleted study: {study_name}")
            return True
        except KeyError:
            self.logger.debug(f"Study not found for deletion: {study_name}")
            return False

    def delete_all_studies(self) -> int:
        """
        Delete all Maestro studies from storage.

        Returns:
            Number of studies deleted
        """
        deleted = 0
        summaries = self.list_studies()

        for summary in summaries:
            if summary.study_name.startswith(self.STUDY_PREFIX):
                try:
                    optuna.delete_study(
                        study_name=summary.study_name,
                        storage=self.storage_url,
                    )
                    deleted += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {summary.study_name}: {e}")

        self.logger.info(f"Deleted {deleted} studies")
        return deleted


# Module-level convenience function for creating studies
def create_study_with_dashboard(
    fold_idx: int,
    storage_url: str | None = None,
    direction: str = "maximize",
    load_if_exists: bool = True,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    study_name_override: str | None = None,
) -> optuna.Study:
    """
    Create an Optuna study with dashboard-compatible SQLite storage.

    Study naming convention (Phase 1):
        maestro_{strategy}_{asset}_fold_{n}
    Falls back to maestro_fold_{n} if no override provided.

    Args:
        fold_idx: Fold index for study naming
        storage_url: SQLite storage URL. Uses default if not provided.
        direction: Optimization direction ("maximize" or "minimize")
        load_if_exists: If True, load existing study for warm-starting.
        sampler: Optuna sampler. Defaults to TPESampler.
        pruner: Optuna pruner. Defaults to MedianPruner.
        study_name_override: Full study name (overrides default naming).

    Returns:
        optuna.Study instance with SQLite storage
    """
    # Determine storage URL
    if storage_url is None:
        storage_url = get_storage_url()

    # Ensure directory exists and enable WAL mode for concurrent writes (Phase 1)
    if storage_url.startswith("sqlite:///"):
        db_path = storage_url[len("sqlite:///"):]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _enable_wal_mode(db_path)

    # Generate study name — use override if provided (Phase 1 naming)
    study_name = study_name_override or f"{OptunaDashboardStorage.STUDY_PREFIX}{fold_idx}"

    # Default sampler with reproducible seed
    if sampler is None:
        sampler = TPESampler(seed=42 + fold_idx)

    # Default pruner
    if pruner is None:
        pruner = MedianPruner(n_startup_trials=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists,
    )

    return study
