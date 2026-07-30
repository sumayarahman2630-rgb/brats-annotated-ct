"""Tests for scripts/train_stage3_watchdog.py: the pure CSV-parsing helpers,
and the slowdown/stall/crash detection logic in run_one_attempt and main(),
exercised against a small FAKE trainer script (not the real, GPU-needing
training script) that can be told to crash, hang, or slow down on command
at a specific step. CPU-only, runs in a few seconds -- every timeout/interval
below is scaled down to test speed, not real training speed.
"""
import csv
import sys
import textwrap

import pytest

from scripts.train_stage3_watchdog import (
    last_logged_step,
    read_train_rows,
    resolve_total_steps,
    run_one_attempt,
)

FAKE_TRAINER = textwrap.dedent('''
    import argparse, csv, os, sys, time

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--total_steps", type=int, required=True)
    parser.add_argument("--fail_mode", default="none", choices=["none", "slowdown", "crash", "hang", "crash_once"])
    parser.add_argument("--fail_at_step", type=int, default=999999)
    parser.add_argument("--step_interval_sec", type=float, default=0.02)
    parser.add_argument("--slow_interval_sec", type=float, default=1.0)
    parser.add_argument("--marker_file", default=None)
    args = parser.parse_args()

    def last_step():
        if not os.path.exists(args.log_file):
            return 0
        best = 0
        with open(args.log_file, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    best = max(best, int(row["step"]))
                except (KeyError, ValueError):
                    pass
        return best

    if not os.path.exists(args.log_file):
        with open(args.log_file, "w", newline="") as f:
            csv.writer(f).writerow(["step", "split", "loss", "dice_score", "lr", "elapsed_sec"])

    step = last_step()
    elapsed = 0.0
    while step < args.total_steps:
        if args.fail_mode == "crash" and step >= args.fail_at_step:
            sys.exit(1)
        if args.fail_mode == "crash_once" and step >= args.fail_at_step and not os.path.exists(args.marker_file):
            open(args.marker_file, "w").close()
            sys.exit(1)
        if args.fail_mode == "hang" and step >= args.fail_at_step:
            time.sleep(3600)
        interval = args.slow_interval_sec if (args.fail_mode == "slowdown" and step >= args.fail_at_step) else args.step_interval_sec
        time.sleep(interval)
        step += 1
        elapsed += interval
        with open(args.log_file, "a", newline="") as f:
            csv.writer(f).writerow([step, "train", 0.1, 0.5, 0.001, f"{elapsed:.4f}"])
    sys.exit(0)
''')


@pytest.fixture
def fake_trainer(tmp_path):
    path = tmp_path / "fake_trainer.py"
    path.write_text(FAKE_TRAINER)
    return path


def _cmd(fake_trainer, log_file, total_steps, **kwargs):
    cmd = [sys.executable, str(fake_trainer), "--log_file", str(log_file), "--total_steps", str(total_steps)]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def test_read_train_rows_filters_to_train_split_only(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "split", "loss", "dice_score", "lr", "elapsed_sec"])
        writer.writerow([25, "train", 0.5, 0.1, 0.001, "10.0"])
        writer.writerow([25, "val", 0.6, 0.05, 0.001, "10.5"])
        writer.writerow([50, "train", 0.4, 0.2, 0.001, "20.0"])
    rows = read_train_rows(str(path))
    assert rows == [(25, 10.0), (50, 20.0)]


def test_read_train_rows_returns_empty_for_missing_file(tmp_path):
    assert read_train_rows(str(tmp_path / "does_not_exist.csv")) == []


def test_read_train_rows_ignores_malformed_trailing_row(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="") as f:
        f.write("step,split,loss,dice_score,lr,elapsed_sec\n")
        f.write("25,train,0.5,0.1,0.001,10.0\n")
        f.write("50,train,0.4,0.2,0.00")  # torn mid-write, no newline, incomplete
    rows = read_train_rows(str(path))
    assert rows == [(25, 10.0)]


def test_last_logged_step_considers_train_and_val_rows(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "split", "loss", "dice_score", "lr", "elapsed_sec"])
        writer.writerow([25, "train", 0.5, 0.1, 0.001, "10.0"])
        writer.writerow([50, "val", 0.6, 0.05, 0.001, "20.5"])
    assert last_logged_step(str(path)) == 50


def test_last_logged_step_is_zero_for_missing_file(tmp_path):
    assert last_logged_step(str(tmp_path / "nope.csv")) == 0


def test_resolve_total_steps_prefers_override():
    config = {"training": {"total_steps": 20000}}
    assert resolve_total_steps(config, None) == 20000
    assert resolve_total_steps(config, 100) == 100


def test_run_one_attempt_succeeds_on_clean_completion(tmp_path, fake_trainer):
    log_file = tmp_path / "log.csv"
    cmd = _cmd(fake_trainer, log_file, total_steps=10, step_interval_sec=0.01)
    success, reason = run_one_attempt(
        cmd, str(log_file), total_steps=10,
        slowdown_multiplier=4.0, stall_timeout_sec=5.0, poll_interval_sec=0.05,
        min_samples_before_trigger=2, grace_period_sec=1.0,
    )
    assert success, reason
    assert last_logged_step(str(log_file)) == 10


def test_run_one_attempt_detects_slowdown_and_stops_the_process(tmp_path, fake_trainer):
    log_file = tmp_path / "log.csv"
    cmd = _cmd(
        fake_trainer, log_file, total_steps=50,
        fail_mode="slowdown", fail_at_step=5, step_interval_sec=0.01, slow_interval_sec=1.0,
    )
    success, reason = run_one_attempt(
        cmd, str(log_file), total_steps=50,
        slowdown_multiplier=4.0, stall_timeout_sec=30.0, poll_interval_sec=0.05,
        min_samples_before_trigger=2, grace_period_sec=1.0,
    )
    assert not success
    assert "slowdown" in reason
    # must not have run all the way to total_steps -- it was killed early
    assert last_logged_step(str(log_file)) < 50


def test_run_one_attempt_detects_a_stalled_process(tmp_path, fake_trainer):
    log_file = tmp_path / "log.csv"
    cmd = _cmd(fake_trainer, log_file, total_steps=50, fail_mode="hang", fail_at_step=3, step_interval_sec=0.01)
    success, reason = run_one_attempt(
        cmd, str(log_file), total_steps=50,
        slowdown_multiplier=4.0, stall_timeout_sec=0.3, poll_interval_sec=0.05,
        min_samples_before_trigger=2, grace_period_sec=1.0,
    )
    assert not success
    assert "hung" in reason or "stall" in reason.lower()


def test_run_one_attempt_detects_a_crash(tmp_path, fake_trainer):
    log_file = tmp_path / "log.csv"
    cmd = _cmd(fake_trainer, log_file, total_steps=50, fail_mode="crash", fail_at_step=3, step_interval_sec=0.01)
    success, reason = run_one_attempt(
        cmd, str(log_file), total_steps=50,
        slowdown_multiplier=4.0, stall_timeout_sec=5.0, poll_interval_sec=0.05,
        min_samples_before_trigger=2, grace_period_sec=1.0,
    )
    assert not success
    assert "exited with code" in reason


def test_run_one_attempt_resumes_from_where_a_previous_attempt_left_off(tmp_path, fake_trainer):
    """The fake trainer (like the real one) reads the log file's own last
    step and continues from there -- confirms run_one_attempt's 'new rows
    since n_rows_before' bookkeeping works correctly across two separate
    launches sharing one log file, the same way a real restart would."""
    log_file = tmp_path / "log.csv"
    cmd1 = _cmd(fake_trainer, log_file, total_steps=5, step_interval_sec=0.01)
    success1, _ = run_one_attempt(
        cmd1, str(log_file), total_steps=5,
        slowdown_multiplier=4.0, stall_timeout_sec=5.0, poll_interval_sec=0.05,
        min_samples_before_trigger=2, grace_period_sec=1.0,
    )
    assert success1
    assert last_logged_step(str(log_file)) == 5

    cmd2 = _cmd(fake_trainer, log_file, total_steps=10, step_interval_sec=0.01)
    success2, _ = run_one_attempt(
        cmd2, str(log_file), total_steps=10,
        slowdown_multiplier=4.0, stall_timeout_sec=5.0, poll_interval_sec=0.05,
        min_samples_before_trigger=2, grace_period_sec=1.0,
    )
    assert success2
    assert last_logged_step(str(log_file)) == 10


@pytest.mark.slow
def test_main_recovers_from_a_single_crash_and_completes(tmp_path, fake_trainer):
    """End-to-end: a fake trainer that crashes exactly once partway through
    must still result in main() reaching total_steps via a restart."""
    log_file = tmp_path / "log.csv"
    marker = tmp_path / "crashed_once.marker"

    # main() hardcodes the real training module -- to exercise the restart
    # loop end-to-end without a GPU, this drives run_one_attempt/last_logged_step
    # in the same loop shape main() uses, with the fake trainer as the command.
    total_steps = 15
    attempt = 0
    while last_logged_step(str(log_file)) < total_steps and attempt < 5:
        cmd = _cmd(
            fake_trainer, log_file, total_steps=total_steps,
            fail_mode="crash_once", fail_at_step=7, step_interval_sec=0.01, marker_file=str(marker),
        )
        success, reason = run_one_attempt(
            cmd, str(log_file), total_steps=total_steps,
            slowdown_multiplier=4.0, stall_timeout_sec=5.0, poll_interval_sec=0.05,
            min_samples_before_trigger=2, grace_period_sec=1.0,
        )
        if success:
            break
        attempt += 1

    assert last_logged_step(str(log_file)) == total_steps
    assert attempt == 1, f"expected exactly one restart, got {attempt}"
