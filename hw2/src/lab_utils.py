"""Utility helpers for the branch-prediction lab notebooks.

These helpers handle everything *around* the predictor — parsing trace files,
running them through a predictor, evaluating accuracy, comparing several
configurations side-by-side, and producing plots.

Students are not expected to read or modify this file.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


Trace = List[Tuple[int, bool]]


# ---------------------------------------------------------------------------
# Trace loading.
# ---------------------------------------------------------------------------


def load_trace(path: str) -> Trace:
    """Load a ``.trace`` file from disk.

    The expected format is one branch per line:

        0x100 T
        0x104 N

    Lines that are blank or start with ``#`` are ignored. Addresses may be
    written in hex (``0x...``) or decimal. The outcome must be ``T``/``t``
    for Taken or ``N``/``n`` for Not Taken.
    """
    trace: Trace = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed trace line: {raw!r}")
            addr_str, outcome = parts[0], parts[1]
            address = int(addr_str, 0)
            outcome = outcome.upper()
            if outcome == "T":
                taken = True
            elif outcome == "N":
                taken = False
            else:
                raise ValueError(f"Bad outcome {outcome!r} in line {raw!r}")
            trace.append((address, taken))
    return trace


# ---------------------------------------------------------------------------
# Running and evaluating.
# ---------------------------------------------------------------------------


def run_trace(predictor, trace: Trace) -> Dict:
    """Run ``trace`` through ``predictor`` and return a results dictionary.

    The dictionary contains the per-step history that the notebook uses for
    plots and analysis:

        addresses        list[int]
        actual           list[bool]
        predicted        list[bool]
        correct          list[bool]
        accuracy         float in [0, 1]
        mispredictions   int
        pht_index_history  list[int]
    """
    addresses: List[int] = []
    actual: List[bool] = []
    predicted: List[bool] = []
    correct: List[bool] = []
    pht_indices: List[int] = []

    cfg = predictor.config
    # Use the helpers from the predictor's own module so that this works
    # both with the solved and the blank implementation once completed.
    import importlib
    helpers = importlib.import_module(type(predictor).__module__)

    for address, taken in trace:
        # Record the PHT index that the predictor *will* consult, based on
        # its current internal history (pre-update).
        history = _current_history(predictor, address)
        if history is None:
            pht_indices.append(-1)
        else:
            pht_indices.append(helpers.get_pht_index(
                address, history, cfg.history_size, cfg.share))

        pred = predictor.predict_and_update(address, taken)
        addresses.append(address)
        actual.append(taken)
        predicted.append(pred)
        correct.append(pred == taken)

    total = max(len(trace), 1)
    mispredictions = total - sum(1 for c in correct if c)
    accuracy = 1 - mispredictions / total

    return {
        "addresses": addresses,
        "actual": actual,
        "predicted": predicted,
        "correct": correct,
        "accuracy": accuracy,
        "mispredictions": mispredictions,
        "total": total,
        "pht_index_history": pht_indices,
    }


def _current_history(predictor, address: int):
    """Return the history register the predictor would consult for ``address``.

    For global mode this is just ``ghr``. For local mode it looks up the
    direct-mapped entry and checks the tag — on a miss we return None to
    signal that the prediction will default to N.
    """
    cfg = predictor.config
    if cfg.history_mode == "global":
        return predictor.ghr
    # Late import to support both solved/blank modules.
    import importlib
    helpers = importlib.import_module(type(predictor).__module__)
    idx = helpers.get_local_index(address, cfg.history_table_size)
    entry = predictor.local_table[idx]
    tag = helpers.get_local_tag(address, cfg.history_table_size)
    if not entry.valid or entry.tag != tag:
        return None
    return entry.bhr


def evaluate_predictor(config, trace_path: str, predictor_module) -> Dict:
    """Build a predictor with ``config``, run a trace, return a summary.

    The returned dict adds ``storage_bits`` and ``trace_name`` to whatever
    ``run_trace`` returns, so it is suitable for tables and plots.
    """
    predictor = predictor_module.BranchPredictor(config)
    trace = load_trace(trace_path)
    res = run_trace(predictor, trace)
    res["storage_bits"] = predictor.storage_bits()
    res["trace_name"] = os.path.basename(trace_path)
    res["config"] = config
    return res


def compare_predictors(configs: Dict[str, "object"],
                       trace_paths: Sequence[str],
                       predictor_module,
                       print_table: bool = True) -> Dict[str, Dict[str, Dict]]:
    """Evaluate every (config, trace) pair and optionally print a table.

    Returns a nested dict ``summary[trace_name][config_name] = results``.
    """
    summary: Dict[str, Dict[str, Dict]] = {}
    for trace_path in trace_paths:
        trace_name = os.path.basename(trace_path)
        summary[trace_name] = {}
        for name, cfg in configs.items():
            summary[trace_name][name] = evaluate_predictor(
                cfg, trace_path, predictor_module)

    if print_table:
        _print_accuracy_table(summary, configs)
    return summary


def _print_accuracy_table(summary, configs):
    config_names = list(configs.keys())
    col_width = max(14, max(len(n) for n in config_names) + 2)
    trace_col = max(20, max(len(t) for t in summary.keys()) + 2)
    header = "trace".ljust(trace_col) + "".join(n.ljust(col_width) for n in config_names)
    print("Accuracy by predictor / trace")
    print(header)
    print("-" * len(header))
    for trace_name, per_cfg in summary.items():
        row = trace_name.ljust(trace_col)
        for name in config_names:
            acc = per_cfg[name]["accuracy"]
            row += f"{acc * 100:6.2f}%".ljust(col_width)
        print(row)

    print("-" * len(header))
    row = "AVERAGE".ljust(trace_col)
    for name in config_names:
        accs = [per_cfg[name]["accuracy"] for per_cfg in summary.values()]
        row += f"{sum(accs) / len(accs) * 100:6.2f}%".ljust(col_width)
    print(row)

    print()
    print("Mispredictions by predictor / trace")
    print(header)
    print("-" * len(header))
    for trace_name, per_cfg in summary.items():
        row = trace_name.ljust(trace_col)
        for name in config_names:
            row += f"{per_cfg[name]['mispredictions']:>6}".ljust(col_width)
        print(row)

    print()
    print("Storage bits by predictor")
    name_col = max(20, max(len(n) for n in config_names) + 2)
    print("predictor".ljust(name_col) + "storage_bits")
    print("-" * (name_col + 12))
    seen = set()
    for trace_name, per_cfg in summary.items():
        for name in config_names:
            if name in seen:
                continue
            seen.add(name)
            print(name.ljust(name_col) + f"{per_cfg[name]['storage_bits']:>10}")


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------


def plot_prediction_timeline(results: Dict, title: str = "Prediction timeline", ax=None):
    """Plot per-step actual vs predicted outcomes.

    Correct predictions are drawn in one color and incorrect ones in another
    so students can spot bursts of mispredictions visually.
    """
    actual = results["actual"]
    correct = results["correct"]
    steps = list(range(len(actual)))
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.6))
    # Outcome line.
    ax.plot(steps, [1 if a else 0 for a in actual], "-",
            color="#777777", linewidth=1, label="actual")
    correct_steps = [s for s, c in zip(steps, correct) if c]
    wrong_steps = [s for s, c in zip(steps, correct) if not c]
    ax.scatter(correct_steps,
               [1 if actual[s] else 0 for s in correct_steps],
               marker="o", color="#2ca02c", s=22, label="correct")
    ax.scatter(wrong_steps,
               [1 if actual[s] else 0 for s in wrong_steps],
               marker="x", color="#d62728", s=40, label="misprediction")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["N", "T"])
    ax.set_xlabel("step")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_counter_evolution(results: Dict, title: str = "2-bit counter evolution", ax=None):
    """Plot how the 2-bit counter state evolves over time.

    ``results`` is expected to contain a ``counter_states`` list (added by
    the notebook helper ``simulate_single_counter``).
    """
    counters = results["counter_states"]
    actual = results["actual"]
    predicted = results.get("predicted", [None] * len(counters))
    steps = list(range(len(counters)))
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(steps, counters, marker="o", color="#1f77b4", label="counter")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["SNT", "WNT", "WT", "ST"])
    ax.set_xlabel("step")
    ax.set_title(title)
    ax.axhline(1.5, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    # Annotate misses.
    for s in steps:
        if predicted[s] is not None and predicted[s] != actual[s]:
            ax.scatter([s], [counters[s]], marker="x", color="#d62728", s=60,
                       zorder=5)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_accuracy_vs_storage(summary: Dict[str, Dict[str, Dict]],
                             title: str = "Accuracy vs storage",
                             ax=None):
    """Scatter plot of accuracy against storage cost.

    For every config we average accuracy across all traces in ``summary``
    and use the (constant) storage bits.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    config_names = []
    for trace_name, per_cfg in summary.items():
        config_names = list(per_cfg.keys())
        break

    for name in config_names:
        accs = [per_cfg[name]["accuracy"]
                for per_cfg in summary.values() if name in per_cfg]
        if not accs:
            continue
        any_trace = next(iter(summary))
        storage = summary[any_trace][name]["storage_bits"]
        avg_acc = sum(accs) / len(accs)
        ax.scatter([storage], [avg_acc * 100], s=80)
        ax.annotate(name, (storage, avg_acc * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("storage [bits]")
    ax.set_ylabel("avg accuracy [%]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_pht_accesses(results: Dict, title: str = "PHT index usage",
                      ax=None):
    """Bar plot of how often each PHT index was accessed.

    Useful for spotting concentrated aliasing.
    """
    indices = [i for i in results["pht_index_history"] if i >= 0]
    counts = Counter(indices)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))
    if not counts:
        ax.text(0.5, 0.5, "no PHT accesses",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return ax
    xs = sorted(counts.keys())
    ys = [counts[i] for i in xs]
    ax.bar(xs, ys, color="#1f77b4")
    ax.set_xlabel("PHT index")
    ax.set_ylabel("accesses")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    return ax


# ---------------------------------------------------------------------------
# Convenience helpers for the notebooks.
# ---------------------------------------------------------------------------


def simulate_single_counter(outcomes: Iterable[bool], predictor_module) -> Dict:
    """Run a single 2-bit saturating counter over ``outcomes`` and trace it.

    Returns a dict with per-step counter values, predictions, actuals.
    """
    counter = 1  # WNT
    predicted = []
    counters = []
    actuals = []
    for taken in outcomes:
        pred = predictor_module.counter_predict(counter)
        predicted.append(pred)
        counters.append(counter)
        actuals.append(taken)
        counter = predictor_module.counter_update(counter, taken)
    correct = [p == a for p, a in zip(predicted, actuals)]
    return {
        "counter_states": counters,
        "predicted": predicted,
        "actual": actuals,
        "correct": correct,
        "accuracy": sum(correct) / max(len(actuals), 1),
    }


def predictor_configs(predictor_module,
                       history_size: int = 4,
                       history_table_size: int = 16) -> Dict:
    """Return the canonical set of five predictor configurations.

    These are the configurations the notebook compares throughout. They all
    share the same history_size and history_table_size so storage/accuracy
    comparisons are meaningful.
    """
    PredictorConfig = predictor_module.PredictorConfig

    return {
        "global-global": PredictorConfig(
            history_size=history_size,
            history_table_size=history_table_size,
            history_mode="global",
            table_mode="global",
            share=False),
        "gshare": PredictorConfig(
            history_size=history_size,
            history_table_size=history_table_size,
            history_mode="global",
            table_mode="global",
            share=True),
        "local-global": PredictorConfig(
            history_size=history_size,
            history_table_size=history_table_size,
            history_mode="local",
            table_mode="global",
            share=False),
        "lshare": PredictorConfig(
            history_size=history_size,
            history_table_size=history_table_size,
            history_mode="local",
            table_mode="global",
            share=True),
        "local-local": PredictorConfig(
            history_size=history_size,
            history_table_size=history_table_size,
            history_mode="local",
            table_mode="local",
            share=False),
    }
