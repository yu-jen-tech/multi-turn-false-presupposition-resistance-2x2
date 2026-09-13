from __future__ import annotations

from pathlib import Path
from typing import Callable

from .client import ChatClient
from .generator import TranscriptGenerator
from .models import (
    BatchGenerationPlan,
    BaselineRecord,
    ChatResult,
    DatasetItem,
    GenerationSettings,
    PushbackRecord,
    Scenario,
)
from .storage import (
    resumable_transcript_exists,
    reusable_baselines,
    reusable_first_response,
    reusable_pushbacks,
    transcript_path,
    write_transcript,
)


ProgressCallback = Callable[[str], None]


def generate_transcripts(
    *,
    plan: BatchGenerationPlan,
    client: ChatClient,
    progress: ProgressCallback = print,
) -> list[Path]:
    """Execute a batch plan and return generated or resumed transcript paths."""

    return _BatchTranscriptGenerator(
        plan=plan,
        client=client,
        progress=progress,
    ).run()


class _BatchTranscriptGenerator:
    def __init__(
        self,
        *,
        plan: BatchGenerationPlan,
        client: ChatClient,
        progress: ProgressCallback,
    ) -> None:
        self._plan = plan
        self._progress = progress
        self._reuse_existing = plan.resume and not plan.overwrite
        self._generators = {
            settings.target_model: TranscriptGenerator(
                client=client,
                settings=settings,
                dataset_provenance=plan.dataset_provenance,
            )
            for settings in plan.settings
        }
        self._first_response_cache: dict[tuple[str, str], ChatResult] = {}

    def run(self) -> list[Path]:
        written: list[Path] = []
        for item in self._plan.items:
            shared_baselines = self._prepare_shared_baselines(item)
            shared_pushbacks = self._prepare_shared_pushbacks(
                item=item,
                shared_baselines=shared_baselines,
            )
            for settings in self._plan.settings:
                pending = self._pending_scenarios(
                    item=item,
                    settings=settings,
                    written=written,
                )
                if not pending:
                    continue
                self._generate_pending_scenarios(
                    item=item,
                    settings=settings,
                    scenarios=pending,
                    shared_pushbacks=shared_pushbacks,
                    written=written,
                )
        return written

    def _prepare_shared_baselines(
        self,
        item: DatasetItem,
    ) -> list[BaselineRecord] | None:
        historyless_scenarios = [
            scenario
            for scenario in self._plan.scenarios
            if scenario.shares_pushbacks_across_target_models
        ]
        if not historyless_scenarios:
            return None
        if not any(
            self._needs_generation(
                item=item,
                scenario=scenario,
                settings=settings,
            )
            for scenario in historyless_scenarios
            for settings in self._plan.settings
        ):
            return None

        baselines = (
            reusable_baselines(
                self._plan.output_dir,
                scenarios=historyless_scenarios,
                item_id=item.id,
            )
            if self._reuse_existing
            else None
        )
        if baselines is not None:
            self._progress(
                f"Reusing shared baselines for {item.id} across styles and target models"
            )
            return baselines

        self._progress(
            f"Generating shared baselines for {item.id} across styles and target models"
        )
        first_target_model = self._plan.settings[0].target_model
        return self._generators[
            first_target_model
        ].generate_without_history_baselines(
            item=item,
            scenario=historyless_scenarios[0],
        )

    def _needs_generation(
        self,
        *,
        item: DatasetItem,
        scenario: Scenario,
        settings: GenerationSettings,
    ) -> bool:
        path = transcript_path(
            self._plan.output_dir,
            scenario=scenario,
            target_model=settings.target_model,
            item_id=item.id,
        )
        return not (
            self._reuse_existing
            and resumable_transcript_exists(
                path,
                dataset_provenance=self._plan.dataset_provenance,
            )
        )

    def _prepare_shared_pushbacks(
        self,
        *,
        item: DatasetItem,
        shared_baselines: list[BaselineRecord] | None,
    ) -> dict[str, list[PushbackRecord]]:
        shared: dict[str, list[PushbackRecord]] = {}
        for scenario in self._plan.scenarios:
            if not scenario.shares_pushbacks_across_target_models:
                continue
            if not any(
                self._needs_generation(
                    item=item,
                    scenario=scenario,
                    settings=settings,
                )
                for settings in self._plan.settings
            ):
                continue
            shared[scenario.name] = self._pushbacks_for(
                item=item,
                scenario=scenario,
                shared_baselines=shared_baselines,
            )
        return shared

    def _pushbacks_for(
        self,
        *,
        item: DatasetItem,
        scenario: Scenario,
        shared_baselines: list[BaselineRecord] | None,
    ) -> list[PushbackRecord]:
        pushbacks = (
            reusable_pushbacks(
                self._plan.output_dir,
                scenario=scenario,
                item_id=item.id,
            )
            if self._reuse_existing
            else None
        )
        if pushbacks is not None:
            self._progress(
                f"Reusing shared pushbacks for {scenario.name} {item.id} "
                "across target models"
            )
            return pushbacks

        self._progress(
            f"Generating shared pushbacks for {scenario.name} {item.id} "
            "across target models"
        )
        first_target_model = self._plan.settings[0].target_model
        if shared_baselines is None:
            raise RuntimeError(
                f"Missing shared baselines for {scenario.name} {item.id}"
            )
        return self._generators[
            first_target_model
        ].generate_without_history_pushbacks(
            item=item,
            scenario=scenario,
            fixed_baselines=shared_baselines,
        )

    def _pending_scenarios(
        self,
        *,
        item: DatasetItem,
        settings: GenerationSettings,
        written: list[Path],
    ) -> list[Scenario]:
        pending: list[Scenario] = []
        for scenario in self._plan.scenarios:
            output_path = transcript_path(
                self._plan.output_dir,
                scenario=scenario,
                target_model=settings.target_model,
                item_id=item.id,
            )
            if self._needs_generation(
                item=item,
                scenario=scenario,
                settings=settings,
            ):
                pending.append(scenario)
                continue
            self._progress(f"Skipping existing {output_path}")
            written.append(output_path)
        return pending

    def _first_response_for(
        self,
        *,
        item: DatasetItem,
        settings: GenerationSettings,
    ) -> ChatResult:
        cache_key = (settings.target_model, item.id)
        if cache_key in self._first_response_cache:
            return self._first_response_cache[cache_key]

        response = (
            reusable_first_response(
                self._plan.output_dir,
                scenarios=self._plan.scenarios,
                target_model=settings.target_model,
                item_id=item.id,
            )
            if self._reuse_existing and self._plan.scenarios
            else None
        )
        if response is None:
            self._progress(
                f"Generating shared first turn for {item.id} "
                f"with {settings.target_model}"
            )
            response = self._generators[
                settings.target_model
            ].generate_first_response(item=item)
        else:
            self._progress(
                f"Reusing shared first turn for {item.id} "
                f"with {settings.target_model}"
            )
        self._first_response_cache[cache_key] = response
        return response

    def _generate_pending_scenarios(
        self,
        *,
        item: DatasetItem,
        settings: GenerationSettings,
        scenarios: list[Scenario],
        shared_pushbacks: dict[str, list[PushbackRecord]],
        written: list[Path],
    ) -> None:
        generator = self._generators[settings.target_model]
        first_response = self._first_response_for(
            item=item,
            settings=settings,
        )
        for scenario in scenarios:
            fixed_pushbacks = self._fixed_pushbacks(
                item=item,
                scenario=scenario,
                shared_pushbacks=shared_pushbacks,
            )
            self._progress(
                f"Generating {scenario.name} for {item.id} "
                f"with {settings.target_model}"
            )
            transcript = generator.generate(
                item=item,
                scenario=scenario,
                first_response=first_response,
                fixed_pushbacks=fixed_pushbacks,
            )
            path = write_transcript(
                transcript,
                self._plan.output_dir,
                overwrite=self._plan.overwrite,
            )
            self._progress(f"Wrote {path}")
            written.append(path)

    @staticmethod
    def _fixed_pushbacks(
        *,
        item: DatasetItem,
        scenario: Scenario,
        shared_pushbacks: dict[str, list[PushbackRecord]],
    ) -> list[PushbackRecord] | None:
        if not scenario.shares_pushbacks_across_target_models:
            return None
        fixed_pushbacks = shared_pushbacks.get(scenario.name)
        if fixed_pushbacks is None:
            raise RuntimeError(
                f"Missing shared pushbacks for {scenario.name} {item.id}"
            )
        return fixed_pushbacks
