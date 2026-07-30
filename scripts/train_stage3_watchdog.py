"""Stage 3 -- fault-tolerant training wrapper; output: the same checkpoint/log
training/train_stage3_segmentation.py produces, kept alive across restarts.

This does not train anything itself -- it repeatedly (re)launches
training/train_stage3_segmentation.py as a subprocess and lets THAT script
do all the real work (training, checkpointing, per-patient loss logging).
This wrapper's only job is deciding when to kill and relaunch it.

Why this exists: a real, unresolved slowdown has shown up twice on Kaggle
(documented in PROJECT_NOTES.md -- first seen 2026-07-24, recurring
2026-07-29/30) where per-step time climbs from a healthy ~10-30s per 25
steps to several hundred seconds per 25 steps, escalating rather than
spiking once. Disabling DataLoader workers entirely (num_workers=0) did
NOT fix it -- it recurred at a similar step count anyway -- so the root
cause is still unknown (a specific slow/corrupted input file, host/GPU
memory pressure, or something at the Kaggle container level are all still
candidates). Rather than wait for a real diagnosis, this watches the
training log for the SYMPTOM -- per-step time rising well above its own
best rate this attempt, or the log going silent entirely -- and kills +
relaunches the moment either shows up. training/train_stage3_segmentation.py
already auto-resumes from the latest checkpoint on every launch, so a
restart costs at most training.checkpoint_interval steps of progress, not
the whole run -- and a fresh process has also, empirically, sometimes
routed around the problem by landing on a different random patient order.

Run as (in place of calling train_stage3_segmentation.py directly):
    python -m scripts.train_stage3_watchdog --config configs/stage3_ct_segmentation.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import time

import yaml

log = logging.getLogger("train_stage3_watchdog")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [WATCHDOG] %(message)s")


def parse_args():
    """--config plus training passthrough flags (max_steps/max_patients/warm_start_checkpoint) and the knobs controlling when a restart triggers."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage3_ct_segmentation.yaml")
    parser.add_argument("--max_steps", type=int, default=None, help="Forwarded to train_stage3_segmentation.py, and used here to know when training is actually done.")
    parser.add_argument("--max_patients", type=int, default=None, help="Forwarded to train_stage3_segmentation.py.")
    parser.add_argument("--warm_start_checkpoint", type=str, default=None, help="Forwarded to train_stage3_segmentation.py -- only ever passed on the FIRST attempt (see main()'s docstring note).")
    parser.add_argument("--slowdown_multiplier", type=float, default=4.0, help="Restart if the current seconds-per-step rate exceeds this many times the best rate seen so far in the current attempt.")
    parser.add_argument("--stall_timeout_sec", type=float, default=900.0, help="Restart if no new row appears in training.log_file for this long -- catches a hang the rate check alone wouldn't (no new rows at all, not just slow ones).")
    parser.add_argument("--poll_interval_sec", type=float, default=15.0, help="How often to check the log file and the subprocess's status.")
    parser.add_argument("--min_samples_before_trigger", type=int, default=8, help="Don't act on the slowdown-rate check until this many train-log rows have been seen in the current attempt -- avoids false triggers on the first few noisy/warmup steps.")
    parser.add_argument("--max_restarts", type=int, default=20, help="Give up after this many restarts without reaching total_steps -- a safety cap for the case where something is fundamentally broken (bad config, bad data) rather than a transient slowdown.")
    parser.add_argument("--grace_period_sec", type=float, default=10.0, help="Seconds to wait after SIGTERM before escalating to SIGKILL when stopping a stuck/slow attempt.")
    return parser.parse_args()


def resolve_total_steps(config: dict, max_steps_override: int | None) -> int:
    """Same override precedence train_stage3_segmentation.py itself applies -- the watchdog needs to agree on what 'done' means."""
    return max_steps_override if max_steps_override is not None else config["training"]["total_steps"]


def read_train_rows(log_file: str) -> list[tuple[int, float]]:
    """Returns (step, elapsed_sec) for every 'train' row currently in
    log_file, in file order. Deliberately re-reads the whole file every
    call rather than tracking a byte offset -- these logs stay small
    (tens of thousands of rows at most) and re-parsing avoids any
    partial-line bookkeeping against a file another process is actively
    appending to."""
    if not os.path.exists(log_file):
        return []
    rows = []
    with open(log_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "train":
                try:
                    rows.append((int(row["step"]), float(row["elapsed_sec"])))
                except (KeyError, ValueError, TypeError):
                    continue  # a torn/partial last line mid-write -- ignore, it'll be complete next poll
    return rows


def last_logged_step(log_file: str) -> int:
    """Highest step number seen in log_file (train or val row) -- 0 if the file doesn't exist yet or has no rows."""
    if not os.path.exists(log_file):
        return 0
    best = 0
    with open(log_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                best = max(best, int(row["step"]))
            except (KeyError, ValueError, TypeError):
                continue
    return best


def stop_process(proc: subprocess.Popen, grace_period_sec: float) -> None:
    """SIGTERM first (lets CUDA/file handles clean up), escalate to SIGKILL if it hasn't exited within grace_period_sec."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_period_sec)
    except subprocess.TimeoutExpired:
        log.warning("Process did not exit within %.0fs of SIGTERM -- sending SIGKILL.", grace_period_sec)
        proc.kill()
        proc.wait()


def run_one_attempt(
    cmd: list[str], log_file: str, total_steps: int,
    slowdown_multiplier: float, stall_timeout_sec: float, poll_interval_sec: float,
    min_samples_before_trigger: int, grace_period_sec: float,
) -> tuple[bool, str]:
    """Launches one training attempt and watches it until it finishes,
    crashes, stalls, or slows down. Returns (finished_successfully, reason)
    -- finished_successfully is True only if the subprocess exited with
    code 0 AND the log shows it actually reached total_steps, not just any
    clean exit.

    The subprocess's own stdout/stderr are left un-redirected (inherited
    from this process) so the wrapped training script's own logs keep
    streaming live to the console/notebook cell exactly as if it had been
    run directly -- this wrapper's own monitoring is entirely separate,
    done by re-reading log_file, not by intercepting the subprocess's
    console output (which would risk a classic pipe-buffer deadlock if
    left undrained).
    """
    log.info("Launching: %s", " ".join(cmd))
    n_rows_before = len(read_train_rows(log_file))
    proc = subprocess.Popen(cmd)

    best_rate = None  # seconds per step, lowest observed so far THIS attempt
    n_new_rows_seen = 0
    last_new_row_time = time.time()

    try:
        while True:
            time.sleep(poll_interval_sec)
            returncode = proc.poll()

            all_rows = read_train_rows(log_file)
            new_rows = all_rows[n_rows_before:]
            # While still running, don't trust the very last row yet -- it might
            # still be mid-write. Once the process has exited, every row is final.
            usable_new_rows = new_rows[:-1] if (returncode is None and new_rows) else new_rows

            if len(usable_new_rows) > n_new_rows_seen:
                last_new_row_time = time.time()
                for i in range(n_new_rows_seen, len(usable_new_rows)):
                    if i == 0:
                        # the first new row of THIS attempt has no same-attempt
                        # predecessor to diff against -- elapsed_sec resets to
                        # near-zero on every fresh launch, so comparing it against
                        # a row from a previous attempt would be meaningless.
                        continue
                    prev_step, prev_elapsed = usable_new_rows[i - 1]
                    step, elapsed = usable_new_rows[i]
                    d_step, d_elapsed = step - prev_step, elapsed - prev_elapsed
                    if d_step <= 0 or d_elapsed < 0:
                        continue
                    rate = d_elapsed / d_step
                    if best_rate is None or rate < best_rate:
                        best_rate = rate
                    elif i >= min_samples_before_trigger and rate > slowdown_multiplier * best_rate:
                        reason = (
                            f"slowdown detected: {rate:.2f}s/step vs this attempt's best "
                            f"{best_rate:.2f}s/step (> {slowdown_multiplier}x) at logged step {step}"
                        )
                        log.warning(reason)
                        stop_process(proc, grace_period_sec)
                        return False, reason
                n_new_rows_seen = len(usable_new_rows)

            if returncode is not None:
                current_step = last_logged_step(log_file)
                if returncode == 0 and current_step >= total_steps:
                    return True, f"subprocess exited cleanly at step {current_step}"
                reason = f"subprocess exited with code {returncode} at logged step {current_step} (target {total_steps})"
                log.warning(reason)
                return False, reason

            if time.time() - last_new_row_time > stall_timeout_sec:
                reason = f"no new log row for over {stall_timeout_sec:.0f}s -- treating as hung"
                log.warning(reason)
                stop_process(proc, grace_period_sec)
                return False, reason
    finally:
        if proc.poll() is None:
            stop_process(proc, grace_period_sec)


def main():
    """Repeatedly launches training/train_stage3_segmentation.py, restarting it whenever run_one_attempt reports a slowdown, stall, or crash, until total_steps is reached or --max_restarts is exhausted."""
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_file = config["training"]["log_file"]
    total_steps = resolve_total_steps(config, args.max_steps)

    base_cmd = [sys.executable, "-m", "training.train_stage3_segmentation", "--config", args.config]
    if args.max_steps is not None:
        base_cmd += ["--max_steps", str(args.max_steps)]
    if args.max_patients is not None:
        base_cmd += ["--max_patients", str(args.max_patients)]

    attempt = 0
    while True:
        current_step = last_logged_step(log_file)
        if current_step >= total_steps:
            log.info("Already at step %d/%d (from a previous attempt) -- nothing to do.", current_step, total_steps)
            return

        cmd = list(base_cmd)
        # --warm_start_checkpoint only makes sense on the very first attempt -- every
        # restart after that must resume normally from whatever the warm-started run
        # itself already saved, or every restart would re-warm-start from the SAME
        # original checkpoint and never make net progress.
        if attempt == 0 and args.warm_start_checkpoint:
            cmd += ["--warm_start_checkpoint", args.warm_start_checkpoint]

        log.info("Attempt %d/%d -- current step %d/%d", attempt + 1, args.max_restarts + 1, current_step, total_steps)
        success, reason = run_one_attempt(
            cmd, log_file, total_steps,
            args.slowdown_multiplier, args.stall_timeout_sec, args.poll_interval_sec,
            args.min_samples_before_trigger, args.grace_period_sec,
        )
        if success:
            log.info("Training finished: %s", reason)
            return

        attempt += 1
        if attempt > args.max_restarts:
            log.error(
                "Gave up after %d restarts without reaching step %d. Last reason: %s. "
                "This many consecutive failures suggests something other than a transient "
                "slowdown -- check the training log and the last checkpoint directly.",
                args.max_restarts, total_steps, reason,
            )
            raise SystemExit(1)
        log.warning("Restarting (attempt %d/%d) -- previous attempt stopped because: %s", attempt + 1, args.max_restarts + 1, reason)


if __name__ == "__main__":
    main()
