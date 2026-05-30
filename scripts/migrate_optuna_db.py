#!/usr/bin/env python3
"""
Phase 1 / 6.1: Migrate Optuna studies DB.

Backs up the current DB and purges stale generic-named studies (maestro_fold_N)
that mix parameters from different strategies. After this, only properly-named
studies (maestro_{strategy}_{asset}_fold_{n}) will exist.

Usage:
    cd ~/Desktop/maestro && PYTHONPATH=backend python scripts/migrate_optuna_db.py
    cd ~/Desktop/maestro && PYTHONPATH=backend python scripts/migrate_optuna_db.py --dry-run
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import optuna

from engine_v2.optuna_dashboard_storage import get_default_storage_path, get_storage_url


def main():
    parser = argparse.ArgumentParser(description='Migrate Optuna studies DB')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without doing it')
    args = parser.parse_args()

    db_path = get_default_storage_path()
    storage_url = get_storage_url(db_path)

    print(f"DB: {db_path}")
    print(f"Storage: {storage_url}")

    # List all studies
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    print(f"\nFound {len(summaries)} studies:")

    stale = []
    keep = []
    for s in summaries:
        # Stale: old generic naming (maestro_fold_N or fold_N)
        is_stale = (
            s.study_name.startswith('maestro_fold_')
            or s.study_name.startswith('fold_')
        )
        if is_stale:
            stale.append(s)
            print(f"  [DELETE] {s.study_name} ({s.n_trials} trials)")
        else:
            keep.append(s)
            print(f"  [KEEP]   {s.study_name} ({s.n_trials} trials)")

    if not stale:
        print("\nNo stale studies to purge. DB is clean.")
        return

    total_stale_trials = sum(s.n_trials for s in stale)
    print(f"\nWill purge: {len(stale)} studies, {total_stale_trials} trials")
    print(f"Will keep:  {len(keep)} studies")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # Backup
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    print(f"\nBackup: {backup_path}")

    # Purge
    for s in stale:
        optuna.delete_study(study_name=s.study_name, storage=storage_url)
        print(f"  Deleted: {s.study_name}")

    # Verify
    remaining = optuna.get_all_study_summaries(storage=storage_url)
    print(f"\nDone. {len(remaining)} studies remaining.")


if __name__ == '__main__':
    main()
