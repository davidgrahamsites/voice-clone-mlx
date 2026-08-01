---
type: module-contract
module: dataset-manifest-rows
sequence: semantic
---

# dataset — DatasetManifest v1 rows

One job: turn an **accepted** alignment row plus a produced clip into a dataset
row, and serialize the collection. This is the last artifact before training,
so everything that reaches it has already been reviewed.

Two modules, split when the row module crossed the ICM size threshold. The seam
is *what a field may be* versus *what a row does*:

| Module | Owns |
|---|---|
| `dataset_row_schema.py` | schema version, split set, audio-property and row/document key sets, `DatasetRowError`, one validator per field kind |
| `dataset_rows.py` | `DatasetRow`, admission, session-exclusivity, serialization — and **re-exports** the public names above |

Callers import everything from `dataset_rows`; the split did not move the
public surface.

## Inputs

- An `AlignmentRow` whose `review_state` is `accepted`.
- A relative `clip_path`, a caller-supplied `clip_sha256`, `audio_properties`,
  a `session_id`, and a `split`.

Do NOT load: audio, models, the filesystem, the network, or a clock.

## Process

1. Refuse anything but an accepted alignment row.
2. Carry the acceptance actor and time across from that row.
3. Validate the clip reference, checksum, audio properties, session, and split.
4. On write or parse, refuse a collection where one session spans two splits.

## Outputs

One JSON **document** (not JSONL, per the contract) with `schema_version` and
`rows`, sorted keys, `ensure_ascii=False`, indented for review.

## Invariants hold whichever path builds a row

`DatasetRow.__post_init__` runs the full field validation, so direct
construction and JSON parsing face exactly the checks the builder does. An
unsigned admission (`accepted_by=""`), a rejected clip decision, a
non-canonical style, an unknown split, a bad checksum, an absolute clip path,
or a **foreign `schema_version`** are all refused at construction — not only in
`build_dataset_row`. Verified against the dataclass directly, not just the
builder.

The version label is validated like every other field. Without that, a row
could be minted claiming a schema whose rules were never applied — every other
value was checked while the label saying *which* rules applied was taken on
trust.

Styles are validated against `shared.styles.CANONICAL_STYLES` — the one home
for that list — on construction *and* on parse. Aliases such as `netural` are
refused: by the time a row reaches the dataset the style was normalized
upstream, and a manifest records canonical values only.

## Two gates inherited, not re-implemented

**Acceptance.** Only `review_state == "accepted"` is admissible; `pending` and
`rejected` raise. There is deliberately no `accepted_by`/`accepted_at`
parameter — the attribution comes from the signed alignment row, so a dataset
row cannot claim an approval that never happened. This mirrors
pending-by-construction in `alignment_rows`: nothing unreviewed reaches
training.

**Clip admission.** Whether a clip is single-speaker and overlap-free is
decided by [`../alignment/overlap_gate.py`](../alignment/overlap_gate.py),
which is **canonical** — it is on `main` and named by the lifecycle contract.

`build_dataset_row` now **requires** that gate's `ClipDecision` as an explicit
argument. Only `status == "accept"` admits; anything else raises, including a
decision with no stated reason — evidence without a reason is not evidence. The
status and reason are recorded on the row as `clip_decision_status` and
`clip_decision_reason`, so a manifest carries proof the gate ran and what it
said.

The rule itself is **not** re-implemented: a test asserts this module defines
neither `SpeakerTurn` nor `decide_clip`.

### The type check is conditional, and why

`alignment/overlap_gate.py` is on `main` but **absent from this worktree**, and
it was outside the write scope, so it could not be added. A hard import would
make this module un-importable here.

The canonical import is therefore attempted; where it resolves, the decision
must be a real `ClipDecision`. Where it does not, the decision is still
required to carry a `status` and `reason`, and the status is checked
regardless — so a look-alike object cannot smuggle a rejected clip through
either way. The isinstance test skips in this checkout and runs on `main`.

**This should be tightened to an unconditional import once the alignment chain
lands on `main`.**

> **Deferred:** `studio_app/audio_acceptance.py` implements the same admission
> rule and is now redundant. Its migration and retirement are deliberately out
> of scope here and remain open; nothing in this module depends on it.

## Session-exclusive splits

Enforced across the collection, not per row — a single row cannot know what the
others do. `rows_to_json` and `parse_dataset_json` both call
`check_session_exclusive_splits`, and the error names the session and both
splits.

This fails loudly because leakage between train and test invalidates every
evaluation number that follows, and it is invisible once the manifest is
written.

## Field rules

| Field | Rule |
|---|---|
| `clip_path`, `alignment_master_audio` | relative; absolute paths and `..` refused |
| `clip_sha256` | exactly 64 lowercase hex characters |
| `split` | one of `train`, `validation`, `test` |
| `style` | canonical, from `shared.styles.CANONICAL_STYLES` |
| `clip_decision_status` | exactly `accept` |
| `clip_decision_reason` | non-blank |
| `audio_properties` | exactly `sample_rate_hz`, `channels`, `bit_depth`, `duration_s` |
| `text` | the accepted row's `expected_text`, verbatim |

`sample_rate_hz`, `channels`, and `bit_depth` are positive integers;
`duration_s` is finite and positive. `bool` is excluded from every numeric
check — it is an `int` subclass, so `True` would otherwise pass as 1.

`audio_properties` is **copied and frozen** into a `MappingProxyType`, so a
caller mutating their dict afterwards cannot change an existing row.

## The checksum is supplied, never computed

Hashing means reading the clip, and that would turn an artifact contract into a
file-touching utility. A test asserts the module imports no `hashlib` and calls
no digest function.

## Human check

1. Before training, open the manifest and confirm every row carries an
   `accepted_by` you recognize and a `clip_decision_reason` that makes sense.
   An unattributed or ungated row should be impossible; if one appears, stop
   and find out how.
2. Check the split assignment by session, not by row — one session's clips must
   all land in the same split.
3. Spot-check a few `clip_sha256` values against the files on disk. This module
   records what it was told; it cannot detect a wrong digest.

## Not implemented

Producing clips, computing checksums, choosing splits, and the rejected-rows
report the contract calls for. Each is a separate seam.

## TDD evidence

```bash
# red — no module yet
$ python3 -m pytest tests/voice_studio/test_dataset_rows.py -q
ERROR — ModuleNotFoundError: No module named
        'voiceclonemlx.dataset.dataset_rows'

# green — the tests live in THREE modules, so all three must be named
$ python3 -m pytest tests/voice_studio/test_dataset_rows.py \
      tests/voice_studio/test_dataset_rows_admission.py \
      tests/voice_studio/test_dataset_rows_gate.py -q
202 passed, 1 skipped
$ git diff --check
(clean)
```

Running only `test_dataset_rows.py` reports **88** and silently skips both
admission and the gate. The figure for this seam is **88 + 53 + 61 = 202**,
plus **one skip, which lives under the gate module** — the `isinstance` check
that needs `alignment/overlap_gate.py` present.

| Run | Result |
|---|---|
| suite **without** this seam (baseline) | `1150 passed, 10 skipped` |
| suite **with** this seam | `1352 passed, 11 skipped` |

Worktree totals only — this checkout carries several other in-flight seams and
is behind `main`; the delta (+202, one skipped) is the figure that travels. The baseline
must ignore **all three** test modules:

```bash
$ python3 -m pytest \
    --ignore=tests/voice_studio/test_dataset_rows.py \
    --ignore=tests/voice_studio/test_dataset_rows_admission.py \
    --ignore=tests/voice_studio/test_dataset_rows_gate.py -q
1150 passed, 10 skipped
```

| Test module | Covers | Tests |
|---|---|---|
| `tests/voice_studio/test_dataset_rows.py` | field rules (clip path, checksum, audio properties, content), the JSON document, the immutable result, purity | 88 |
| `tests/voice_studio/test_dataset_rows_admission.py` | admission of accepted alignment rows and refusal of pending/rejected, attribution carry-over, split vocabulary, session exclusivity on write and parse, full re-validation of a hand-edited manifest | 53 |
| `tests/voice_studio/test_dataset_rows_gate.py` | the required clip decision and its recorded evidence, and that neither direct construction nor a hand-edited manifest can bypass any invariant | 61 (+1 skipped) |

The three test modules mirror the two production modules: shape, admission,
and gate.

## Depends on unmerged work

`alignment_rows.py` and `alignment_row_schema.py` are **not on `main`** — they
exist only in this worktree, as does everything else in the alignment chain
except `marker_parser` and `overlap_gate`. This module cannot be integrated
before them, and its tests will not collect in a `main` checkout.

## Split planning

`split_plan.py` answers the question `dataset_rows` only validates: *which*
split a session belongs to. `plan_splits(session_ids, *, ratios=SPLIT_RATIOS)`
returns a frozen `SplitPlan` carrying `assignments`, `counts`, and the `ratios`
that were asked for; `split_for(plan, session_id)` looks one up and raises on
an unknown session rather than defaulting to `train` — a silent default is
exactly how a test clip reaches the training set.

Assignment is **sorted largest-remainder**, not hashed. Ids are validated,
refused if blank, non-string, or duplicated, then sorted by bytes. Each split
takes the whole part of its share; leftover seats go one each to the largest
fractional remainders, ties broken by `SPLITS` order. Sessions then fill
`train`, `validation`, `test` in sorted order. A hashed assignment would be
just as stable, but nobody can verify it by reading the manifest — this one is
checkable with a pencil.

Small corpora land **entirely in `train`**: at 80/10/10, one, two, and three
sessions all go to `train`; `validation` first appears at four and `test` at
seven. That is the honest outcome — a one-clip test split produces a number
nobody should quote.

`SplitPlan.__post_init__` applies every builder rule, so direct construction
faces the checks `plan_splits` does: exact split keys on `counts` and `ratios`,
finite positive ratios summing to one, non-negative integer counts (`bool`
excluded), assignments naming only real splits with non-blank string ids, and
`counts` agreeing with the assignments they summarize. A rule only the builder
enforced would be no rule at all — `SplitPlan` is public, and a hand-built plan
claiming test sessions it never assigned would reshape a dataset manifest.

Session exclusivity is structural: each id is assigned once, so a plan-derived
manifest cannot trip `check_session_exclusive_splits`. That check stays the
authority for manifests built any other way and is not restated here. The split
vocabulary is imported from `dataset_rows.SPLITS`, so it keeps one home.

Pure: no hashing, randomness, clock, filesystem, network, or model import.
`bool` is excluded from every ratio check, and ratios may drift from 1.0 by
`RATIO_TOLERANCE` so three thirds are accepted.

TDD evidence (measured at this worktree):

    focused: PYTHONPATH=src python3 -m pytest \
      tests/voice_studio/test_split_plan.py \
      tests/voice_studio/test_split_plan_safety.py -q
      137 passed (50 + 87)
    baseline (both ignored): 1715 passed, 9 skipped
    full suite:              1852 passed, 9 skipped

## Clip probe

`clip_probe.py` supplies the two things `build_dataset_row` requires and never
computes: the clip's SHA-256 and its audio properties. `probe_clip(path, *,
max_bytes=MAX_CLIP_BYTES)` returns a frozen `ClipMeasurement(clip_sha256,
audio_properties, byte_size)` whose keys are exactly `AUDIO_FIELDS`. Keeping
the digest out of `dataset_rows` keeps that contract free of file access.

Values come from the file: the digest from its bytes, the properties from the
`wave` header. `duration_s` is frames over frame rate — frames count across
channels, so stereo is not twice as long as mono. A zero-frame WAV is
**refused**: `dataset_row_schema` requires `duration_s > 0`. Every
`stat`/`open`/read failure surfaces as a chained `ClipProbeError`.

The clip is opened **once**, with `O_NOFOLLOW`: checking `is_symlink()` then
opening by name leaves a window the kernel can close. Size, digest, header, and
regular-file status all come from that descriptor; without the flag, `lstat` is
compared against it — detection, not prevention. The cap is enforced from the descriptor *and* counted while
streaming. Identity (dev, inode, size, mtime, **ctime**) is re-checked after,
and so is the **name**: `os.replace` leaves the read intact, so a true digest
of bytes the path no longer refers to would otherwise be recorded.
`ClipMeasurement.__post_init__` validates direct construction too, `bool` is
excluded from every numeric check, and the properties mapping is copied then
frozen. Stdlib only: `wave`, `hashlib`, `os`, `stat`, `errno`, `pathlib`.

> **Known duplication:** `tts/backend.py` computes the same `frames /
> frame_rate` arithmetic; importing it would mean `dataset` depending on `tts`
> and raising `SynthesisError`. A shared WAV reader in `shared/` remains open.

TDD evidence (measured at this worktree):

    red (no module): ModuleNotFoundError on both modules
    red (race):      2 failed — path resolved three times, so a clip swapped
        mid-probe was measured without complaint
    red (ctime):     same-size rewrite passed without `st_ctime_ns` — vacuous
        until the mtime was restored too
    red (rename):    a renamed replacement was measured silently; the name is
        now re-checked with `lstat` against the descriptor
    green:           137 passed (75 + 42 + 20, across three modules)
    baseline (ignored):  1852 passed, 9 skipped
    full suite:          1989 passed, 9 skipped
