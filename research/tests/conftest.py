"""Skips for the public copy of the test suite.

Some tests check artefacts that are not part of this folder: the training data (built by
scripts/train/make_data_v*.sh, not redistributed), raw run records (runs/), per-host records of the cloud runs, the
paper's LaTeX sources, or the exact bytes of files that were sanitized for publication. Those tests are skipped here
with the reason; tests that need an optional package run once it is installed.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

NEEDS_MODULE = {
    "torch": [
        "tests/test_e14_clef.py::test_served_through_serve_jev_build_app",
        "tests/test_explore_e12.py::test_items_render_as_served_and_e1_format",
        "tests/test_explore_e28.py::test_merge_vendor_keeps_the_layout_and_adds_scaled_lora",
        "tests/test_explore_e31.py::test_training_config_differs_from_v1_only_as_registered",
        "tests/test_tournament_passes.py::test_jev_client_reads_server_reported_passes_and_hosted_jev_stays_unobserved",
        "tests/test_training_kit.py::test_harvested_tool_options_match_the_served_prompt",
    ],
    "alfworld": ["tests/test_explore_e24.py::"],
}
NEEDS_PATH = {
    "data/train_v0/train.jsonl": [
        "tests/test_training_kit.py::test_smoke_gpu_defaults_to_the_small_model",
        "tests/test_training_kit.py::test_train_v0_has_no_eval_items_and_valid_targets",
        "tests/test_training_kit.py::test_train_v0_manifest_matches_files_and_size_limit",
        "tests/test_training_kit.py::test_train_v0_trajectories_are_training_seed_shadow_runs",
        "tests/test_training_data_v1.py::test_gate_dev_tasks_and_e1_ids_load_from_the_repo_files",
        "tests/test_training_data_v1.py::test_12_sft_v1_dry_run_trains_evaluates_and_continues_with_opd",
        "tests/test_recipe_takeaways.py::",
    ],
    "data/train_v1/train.jsonl.gz": [
        "tests/test_e14_clef.py::test_sft_v1_dev_file_converts_completely",
        "tests/test_explore_e7d.py::test_harvest_and_build_keep_heldout_apart",
        "tests/test_explore_e27.py::test_argument_rows_of_a_step_stay_one_record",
        "tests/test_explore_e27.py::test_augmentation_renames_consistently_and_moves_only_the_label",
        "tests/test_explore_e27.py::test_selection_is_half_of_the_states_and_deterministic",
        "tests/test_explore_e28.py::test_build_augments_agent_states_only_and_keeps_the_ratio",
        "tests/test_explore_e28.py::test_build_refuses_a_different_mix_and_unknown_rows",
        "tests/test_explore_e28.py::test_build_refuses_guarded_seeds",
        "tests/test_stage2_v2.py::test_v2_plan_matches_the_freeze_record",
    ],
    "runs": ["tests/test_e17_e18.py::test_committed_head_profile_and_live_configs"],
    "results/stage2/s23/answers_sft_v1.jsonl": [
        "tests/test_e5_small.py::test_analysis_reproduces_the_published_rows_and_applies_the_rules",
        "tests/test_e5_small.py::test_compact_bench_keeps_what_the_analysis_reads",
    ],
    "results/stage2/s23/answers_rl.jsonl": [   # the models' answers on corpus C are not redistributed
        "tests/test_benchmark_difficulty.py::test_corpus_strata_and_registered_accuracies",
        "tests/test_coverage_ask_act.py::test_committed_corpus_reproduces_published_rows",
    ],
    "results/stage2/gpu": ["tests/test_train_ablation.py::test_anchor_check_tolerances_and_fallback_code"],
    "results/explore/gpu": ["tests/test_explore_e28.py::test_base_revision_is_the_logged_rows"],
    "results/explore/e19_hard_random/box": [
        "tests/test_explore_e19.py::test_e19_configs_run_the_random_rules_checkpoint_on_e7f_tasks_at_the_fitted_thresholds"],
    "results/explore/e7_hard/box_e7d": [
        "tests/test_explore_e7.py::test_e7f_configs_run_the_tier_checkpoint_in_the_e7e_setting_on_fresh_seeds"],
    "results/explore/e11_alfworld/box": ["tests/test_explore_e23.py::test_verify_ckpt_against_e11_record"],
    "results/explore/e21_shift/c/questions.jsonl.gz": [
        "tests/test_explore_e27.py::test_vocabulary_disjoint_from_e21_and_its_rerenderings"],
    "paper/sections": ["tests/test_case_studies.py::test_every_key_the_paper_reads_exists"],
    "paper/figures": ["tests/test_explore_e9.py::test_every_key_the_figure_reads_is_generated"],
    "docs/TRAINING_AGENT_BRIEF.md": [
        "tests/test_training_kit.py::test_brief_references_existing_scripts_and_states_hard_rules"],
}
SANITIZED = [   # tests of the exact cloud drivers and sha256 records, which the public copy rewrites
    "tests/test_e17_e18.py::test_driver_syntax_and_never_touch_list",
    "tests/test_e5_small.py::test_driver_live_dry_run",
    "tests/test_e5_small.py::test_driver_part_host_dry_run_and_analysis_merge",
    "tests/test_e5_small.py::test_driver_syntax_refusals_and_offline_dry_run",
    "tests/test_explore_e11.py::test_driver_syntax_and_forbidden_hosts",
    "tests/test_explore_e21b.py::test_driver_syntax_and_guards",
    "tests/test_explore_e21b.py::test_e12_profile_pending_or_fitted",
    "tests/test_explore_e23.py::test_driver_syntax_and_forbidden_hosts",
    "tests/test_explore_e27.py::test_driver_syntax_and_guards",
    "tests/test_explore_e27.py::test_profile_pending_or_fitted",
    "tests/test_explore_e28.py::test_driver_syntax_and_guards",
    "tests/test_explore_e28.py::test_profile_pending_or_fitted",
    "tests/test_explore_e30.py::test_registration_and_driver_guards",
    "tests/test_explore_e31.py::test_driver_syntax_and_guards",
    "tests/test_train_ablation.py::test_driver_syntax_and_dry_run_one_arm",
]


def _match(nodeid: str, prefixes) -> bool:
    base = nodeid.split("[", 1)[0]
    return any(base == p or (p.endswith("::") and base.startswith(p)) for p in prefixes)


def pytest_collection_modifyitems(config, items):
    for item in items:
        nid = item.nodeid
        if not nid.startswith("tests/"):
            nid = "tests/" + nid.split("tests/", 1)[-1]
        if _match(nid, SANITIZED):
            item.add_marker(pytest.mark.skip(reason="checks the exact private cloud driver or a sanitized sha256 record"))
            continue
        for mod, ids in NEEDS_MODULE.items():
            if _match(nid, ids) and importlib.util.find_spec(mod) is None:
                item.add_marker(pytest.mark.skip(reason=f"needs the optional package {mod}"))
        for path, ids in NEEDS_PATH.items():
            if _match(nid, ids) and not (ROOT / path).exists():
                item.add_marker(pytest.mark.skip(reason=f"needs {path}, which is not part of the public folder"))
