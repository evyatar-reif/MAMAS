"""Student skeleton for the configurable 2-level branch predictor.

This module exposes the full predictor API. The infrastructure for
predict-then-update, table management, and tag checking is already wired up
for you; your job is to fill in the functions marked ``# YOUR CODE:``.

Rules:
- Do **not** change function signatures (names, parameters, return values).
- Do **not** change the public API of ``BranchPredictor``.
- You may (and should) call the helper functions that are already provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, log2
from typing import List, Optional


# Counter state encoding.
SNT = 0  # Strongly Not Taken
WNT = 1  # Weakly Not Taken
WT = 2   # Weakly Taken
ST = 3   # Strongly Taken

# Initial value for any fresh 2-bit counter.
COUNTER_INIT = WNT


# ---------------------------------------------------------------------------
# Module-level helper functions that YOU need to complete.
# ---------------------------------------------------------------------------


def counter_predict(counter: int) -> bool:
    # YOUR CODE:
    # Given a 2-bit saturating counter value (0..3) return its prediction.
    # Encoding: SNT=0, WNT=1, WT=2, ST=3.
    # - counter == 0 or 1  ->  Not Taken (return False)
    # - counter == 2 or 3  ->  Taken     (return True)
    match counter:
        case 0 | 1:
            return False
        case 2 | 3:
            return True


def counter_update(counter: int, taken: bool) -> int:
    # YOUR CODE:
    # Given a 2-bit saturating counter and the actual branch outcome, return
    # the new counter value, clamped to [0, 3].
    # - if taken is True  -> increment but do not exceed 3
    # - if taken is False -> decrement but do not go below 0
    delta = 1 if taken else -1
    match counter:
        case 1 | 2:
            return counter + delta
        case 3:
            return min(3 + delta, 3)
        case 0:
            return max(0, delta)



def update_history(history: int, taken: bool, history_size: int) -> int:
    # YOUR CODE:
    # Update a history register (BHR or GHR) by shifting it left and
    # inserting the new outcome bit: 1 if taken, 0 otherwise. Keep only the
    # low ``history_size`` bits of the result.
    # Example: history=0b101, taken=True, history_size=4 -> 0b1011.

    history = history << 1
    if taken:
        history = history | 1

    mask = (1 << history_size) - 1
    return history & mask


def get_local_index(address: int, history_table_size: int) -> int:
    # YOUR CODE:
    # Compute the direct-mapped index into the local history table.
    # ``history_table_size`` is the number of entries and is always a power
    # of two. Return the low bits of ``address`` so the result fits the
    # table size.

    num_of_bits = int(log2(history_table_size))
    mask = (1 << num_of_bits) - 1
    return address & mask

def get_local_tag(address: int, history_table_size: int) -> int:
    # YOUR CODE:
    # Compute the tag stored alongside a local history entry. The tag is
    # the address bits ABOVE the index bits. Use log2 of
    # ``history_table_size`` to know how many bits to shift right.

    tag = address >> int(log2(history_table_size))
    return tag



def get_pht_index(address: int, history: int, history_size: int, share: bool) -> int:
    # YOUR CODE:
    # Compute the index into the PHT (counter table).
    # - If share is False: return the history (masked to history_size bits).
    # - If share is True (gshare/lshare): XOR the history with the low
    #   ``history_size`` bits of ``address``.

    mask = (1 << history_size) - 1
    history = history & mask
    
    if not share:
        return history
    
    address_low = address & mask
    return history ^ address_low



# ---------------------------------------------------------------------------
# Configuration object — already implemented for you. Do not modify.
# ---------------------------------------------------------------------------


@dataclass
class PredictorConfig:
    """Configuration for the 2-level branch predictor.

    Attributes:
        history_size: Number of bits in each history register (BHR/GHR).
        history_table_size: Number of entries in the local history table
            (power of 2). Relevant only when ``history_mode == "local"``.
        history_mode: ``"local"`` or ``"global"``.
        table_mode: ``"local"`` or ``"global"``. ``"local"`` is only valid
            when ``history_mode == "local"``.
        share: When True, XOR low address bits into the PHT index
            (gshare/lshare).
        address_bits: Total width of a branch address in bits.
    """

    history_size: int
    history_table_size: int
    history_mode: str  # "local" or "global"
    table_mode: str    # "local" or "global"
    share: bool
    address_bits: int = 32

    def __post_init__(self) -> None:
        if self.history_mode not in ("local", "global"):
            raise ValueError("history_mode must be 'local' or 'global'")
        if self.table_mode not in ("local", "global"):
            raise ValueError("table_mode must be 'local' or 'global'")
        if self.history_mode == "global" and self.table_mode == "local":
            raise ValueError(
                "global history with local tables is not a supported "
                "configuration"
            )
        if self.history_mode == "local":
            if self.history_table_size < 1:
                raise ValueError("history_table_size must be >= 1")
            if self.history_table_size & (self.history_table_size - 1):
                raise ValueError("history_table_size must be a power of 2")


# ---------------------------------------------------------------------------
# The predictor class. Most of the wiring is provided for you. You only need
# to complete ``predict`` and ``update`` below.
# ---------------------------------------------------------------------------


class _LocalEntry:
    """A single entry in the local history table."""

    __slots__ = ("valid", "tag", "bhr", "pht")

    def __init__(self, has_local_pht: bool, history_size: int) -> None:
        self.valid = False
        self.tag = 0
        self.bhr = 0
        if has_local_pht:
            self.pht: Optional[List[int]] = [COUNTER_INIT] * (1 << history_size)
        else:
            self.pht = None


class BranchPredictor:
    """A configurable 2-level branch predictor."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self._has_local_history = config.history_mode == "local"
        self._has_local_pht = config.table_mode == "local"
        self.reset()

    # ------------------------------------------------------------------
    # State management (provided).
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all predictor state to its initial values."""
        cfg = self.config
        self.ghr: int = 0
        if cfg.history_mode == "local":
            self.local_table: List[_LocalEntry] = [
                _LocalEntry(self._has_local_pht, cfg.history_size)
                for _ in range(cfg.history_table_size)
            ]
        else:
            self.local_table = []
        if not self._has_local_pht:
            self.global_pht: List[int] = [COUNTER_INIT] * (1 << cfg.history_size)
        else:
            self.global_pht = []

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def predict(self, address: int) -> bool:
        # YOUR CODE:
        # Return the prediction for ``address`` WITHOUT updating any state.
        # Recommended flow:
        # 1) Use ``self._lookup(address, allocate=False)`` to obtain
        #    ``(history, pht, entry)``.
        # 2) If ``pht`` is None (tag miss in local-history mode) return False
        #    (default Not-Taken prediction).
        # 3) Otherwise compute the PHT index with ``get_pht_index`` and
        #    return ``counter_predict(pht[idx])``.

        (history, pht, entry) = self._lookup(address, False)
        if pht is None:
            return False
        
        idx = get_pht_index(address, history, self.config.history_size, self.config.share)
        return counter_predict(pht[idx])


    def update(self, address: int, taken: bool) -> None:
        # YOUR CODE:
        # Update the internal state given the actual outcome.
        # The order of operations matters:
        # 1) ``self._lookup(address, allocate=True)`` — this also handles
        #    tag-mismatch resets automatically.
        # 2) Compute the PHT index from the history BEFORE updating it.
        # 3) Update the counter at pht[idx] via counter_update.
        # 4) Compute the new history with update_history and store it in
        #    the correct place: self.ghr in global mode, entry.bhr in local
        #    mode.

        (history, pht, entry) = self._lookup(address, True)
        idx = get_pht_index(address, history, self.config.history_size, self.config.share)
        pht[idx] = counter_update(pht[idx], taken)


        new_history = update_history(history, taken, self.config.history_size)
        if (self._has_local_history):
            entry.bhr = new_history
            return
        
        self.ghr = new_history


    def predict_and_update(self, address: int, taken: bool) -> bool:
        """Predict, then update — this guarantees predict-before-update order."""
        prediction = self.predict(address)
        self.update(address, taken)
        return prediction

    def storage_bits(self) -> int:
        """Total storage cost of this predictor configuration, in bits."""
        cfg = self.config
        hsize = cfg.history_size
        pht_bits_per_table = 2 * (1 << hsize)

        if cfg.history_mode == "global":
            return hsize + pht_bits_per_table

        index_bits = (cfg.history_table_size - 1).bit_length() if cfg.history_table_size > 1 else 0
        tag_bits = max(cfg.address_bits - index_bits, 0)
        per_entry = 1 + tag_bits + hsize
        if cfg.table_mode == "local":
            per_entry += pht_bits_per_table
            return cfg.history_table_size * per_entry
        return cfg.history_table_size * per_entry + pht_bits_per_table

    # ------------------------------------------------------------------
    # Internals (provided).
    # ------------------------------------------------------------------

    def _lookup(self, address: int, allocate: bool):
        """Resolve the (history, pht, entry) triple for ``address``.

        When ``allocate`` is True, on a tag-mismatch the entry is reset to a
        clean state (BHR=0, PHT=WNT in local-local) using the new tag.
        When ``allocate`` is False (prediction path), a tag-mismatch returns
        ``(0, None, entry)`` so that the caller defaults to N.
        """
        cfg = self.config
        if cfg.history_mode == "global":
            return self.ghr, self.global_pht, None

        idx = get_local_index(address, cfg.history_table_size)
        entry = self.local_table[idx]
        tag = get_local_tag(address, cfg.history_table_size)

        if not entry.valid or entry.tag != tag:
            if not allocate:
                return 0, None, entry
            entry.valid = True
            entry.tag = tag
            entry.bhr = 0
            if self._has_local_pht:
                entry.pht = [COUNTER_INIT] * (1 << cfg.history_size)

        if self._has_local_pht:
            return entry.bhr, entry.pht, entry
        return entry.bhr, self.global_pht, entry
