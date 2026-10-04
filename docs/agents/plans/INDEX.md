# Agent Plan Registry (compatibility view)

Active task state is owned by the v6 records under
`docs/agents/runs/current/<task-id>/task.yaml`. This file is generated for old
links and readers; edit the task record and rerun
`python scripts/render_agent_docs.py --write` instead of editing this view.

<!-- BEGIN GENERATED: current_task_registry -->
| task_id | state | route | cost | shape | condition | record |
| --- | --- | --- | --- | --- | --- | --- |
| `docs-build-skill` | `reviewing` | `modify_code` | `strict` | `staged` | `—` | `docs/agents/runs/current/docs-build-skill/task.yaml` |
| `get-data-output-unification` | `planning` | `modify_code` | `standard` | `staged` | `NeedsRevalidation` | `docs/agents/runs/current/get-data-output-unification/task.yaml` |
| `hit-merged-peaklet-responsibility` | `planning` | `modify_code` | `standard` | `staged` | `NeedsRevalidation` | `docs/agents/runs/current/hit-merged-peaklet-responsibility/task.yaml` |
| `hit-threshold-ragged-optimization` | `planning` | `modify_plugin` | `standard` | `staged` | `NeedsRevalidation` | `docs/agents/runs/current/hit-threshold-ragged-optimization/task.yaml` |
| `import-topology-optimization` | `executing` | `modify_code` | `strict` | `staged` | `ReworkExecutionStarted` | `docs/agents/runs/current/import-topology-optimization/task.yaml` |
| `managed-workflow-mvp` | `completed` | `modify_code` | `strict` | `staged` | `—` | `docs/agents/runs/current/managed-workflow-mvp/task.yaml` |
| `reconstruction-review-fixes` | `completed` | `modify_plugin` | `strict` | `staged` | `—` | `docs/agents/runs/current/reconstruction-review-fixes/task.yaml` |
| `release-v2.0.0` | `completed` | `release_artifact_sync` | `strict` | `staged` | `—` | `docs/agents/runs/current/release-v2.0.0/task.yaml` |
| `v1725-merged-parts-cleanup` | `completed` | `modify_code` | `strict` | `staged` | `—` | `docs/agents/runs/current/v1725-merged-parts-cleanup/task.yaml` |
| `waveform-doc-content-recovery-20260901` | `completed` | `modify_code` | `standard` | `staged` | `—` | `docs/agents/runs/current/waveform-doc-content-recovery-20260901/task.yaml` |
| `waveform-doc-lineage-density-20260901` | `completed` | `modify_code` | `standard` | `staged` | `AllBlockingGatesPass` | `docs/agents/runs/current/waveform-doc-lineage-density-20260901/task.yaml` |
| `waveform-doc-nested-nav-20260902` | `completed` | `modify_code` | `strict` | `staged` | `—` | `docs/agents/runs/current/waveform-doc-nested-nav-20260902/task.yaml` |
| `waveform-doc-review-fixes-20260907` | `completed` | `modify_code` | `strict` | `staged` | `—` | `docs/agents/runs/current/waveform-doc-review-fixes-20260907/task.yaml` |
| `waveform-doc-sidebar-expand-20260902` | `completed` | `modify_code` | `standard` | `staged` | `—` | `docs/agents/runs/current/waveform-doc-sidebar-expand-20260902/task.yaml` |
| `waveform-doc-strict-optimization-20260902` | `completed` | `modify_code` | `strict` | `staged` | `—` | `docs/agents/runs/current/waveform-doc-strict-optimization-20260902/task.yaml` |
| `waveform-doc-system-opt-20260831` | `completed` | `retire_compat` | `strict` | `staged` | `PlanBriefReadyWithUserConfirmation` | `docs/agents/runs/current/waveform-doc-system-opt-20260831/task.yaml` |
| `waveform-version-sync-20260902` | `blocked` | `release_artifact_sync` | `strict` | `staged` | `FirefoxHeadlessEnvironmentBlocked` | `docs/agents/runs/current/waveform-version-sync-20260902/task.yaml` |
<!-- END GENERATED: current_task_registry -->

Legacy plan text is retained as `legacy-plan.md` beside each current task. It is
historical input and is not a second source of truth.
