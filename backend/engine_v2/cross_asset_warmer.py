"""
Phase 3: Cross-Asset Warm-Starting for Optuna Studies

Seeds a new study with the top-N trial params from the same strategy
on different assets. Uses study.enqueue_trial() so TPE treats them as
high-priority starting points without biasing the search.

Only seeds from same-strategy studies. Never cross-strategy.

Example:
    warmer = CrossAssetWarmer(storage_url='sqlite:///data/optuna_studies.db')
    warmer.seed_study(study, strategy_name='EWMAC', current_asset='FTM', top_n=5)
"""

import logging
from typing import Any

import optuna


class CrossAssetWarmer:
    """Seeds Optuna studies with best params from same strategy on other assets."""

    def __init__(
        self,
        storage_url: str,
        logger: logging.Logger | None = None,
    ):
        self.storage_url = storage_url
        self.logger = logger or logging.getLogger(__name__)

    def seed_study(
        self,
        study: optuna.Study,
        strategy_name: str,
        current_asset: str,
        top_n: int = 5,
    ) -> int:
        """Seed a study with best params from same strategy on other assets.

        Args:
            study: Target study to seed
            strategy_name: Strategy identifier (must match study naming convention)
            current_asset: Asset being optimized (excluded from donor studies)
            top_n: Number of best trials to seed per donor study

        Returns:
            Number of trials enqueued
        """
        prefix = f"maestro_{strategy_name}_"
        exclude = f"maestro_{strategy_name}_{current_asset}_"

        try:
            summaries = optuna.get_all_study_summaries(storage=self.storage_url)
        except Exception as e:
            self.logger.warning(f"Failed to list studies for warm-starting: {e}")
            return 0

        # Find donor studies: same strategy, different asset
        donor_names = [
            s.study_name for s in summaries
            if s.study_name.startswith(prefix)
            and not s.study_name.startswith(exclude)
            and s.n_trials > 0
        ]

        if not donor_names:
            self.logger.debug(f"No donor studies found for {strategy_name} (excluding {current_asset})")
            return 0

        enqueued = 0
        seen_params = set()

        for donor_name in donor_names:
            try:
                donor = optuna.load_study(study_name=donor_name, storage=self.storage_url)
            except Exception:
                continue

            # Get top-N completed trials by value
            completed = [
                t for t in donor.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
            ]
            completed.sort(key=lambda t: t.value, reverse=True)

            for trial in completed[:top_n]:
                # Deduplicate by param hash
                param_key = tuple(sorted(trial.params.items()))
                if param_key in seen_params:
                    continue
                seen_params.add(param_key)

                study.enqueue_trial(
                    trial.params,
                    user_attrs={"source": donor_name},
                    skip_if_exists=True,
                )
                enqueued += 1

        if enqueued > 0:
            self.logger.info(
                f"Warm-started {study.study_name} with {enqueued} trials "
                f"from {len(donor_names)} donor studies"
            )

        return enqueued
