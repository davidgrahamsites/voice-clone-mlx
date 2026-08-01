---
type: architecture-contract
contract: voice-model-lifecycle
contract_version: 1.0.0
status: accepted-for-scaffolding
software_release: 0.8.0
version_impact: minor
migration_required: false
usage_scope: personal-noncommercial
owners:
  producer: Voice Studio
  consumer: Voice Reader
---

# Voice model lifecycle

One job: define the stable, versioned handoff by which Voice Studio produces a
loadable personal voice model and Voice Reader consumes it.

## ICM contract

### Inputs

- Product plan: [`../../PLAN.md`](../../PLAN.md)
- Backend decision: [`../research/tts-backend-evaluation.md`](../research/tts-backend-evaluation.md)
- Upstream recording-script handoff: a path to a reviewed JSONL manifest; its
  current implementation and App Opener Hub bus envelope remain outside this
  lifecycle contract.

Do NOT load: application UI, other run folders, prior training runs, or backend
weights that are not referenced by the selected manifest.

### Process

1. Voice Studio advances editable artifacts through recording, transcription,
   alignment, CPU audio preprocessing, dataset construction, remote-CUDA
   training, source evaluation/promotion, MLX conversion, runtime parity
   evaluation, and bundle publication.
2. Source promotion freezes one accepted training checkpoint as a provider-
   neutral `SourceModelRelease`. Conversion creates a separate runtime variant;
   it never replaces or relabels the promoted source checkpoint.
3. Bundle publication copies the source release, accepted runtime variants,
   references, licenses, and evidence into a new immutable `VoiceModelBundle`.
4. Voice Reader opens the promoted bundle only through `VoiceModelLoading`, then
   generates through `SpeechSynthesizing` using an explicit execution policy.

### Output

- One immutable `VoiceModelBundle` directory matching Bundle Contract v1.

### Human check

Before publishing `promoted.json`, open the source and runtime evaluation
reports, listen to every acceptance-set sample, and approve the exact source-
release checksum, runtime-variant checksum, bundle id/revision/checksum, and
intended usage. Voice Reader never approves or promotes models.

## Architectural decision

VoiceCloneGPT is an umbrella over two independently deletable local apps:

```text
Voice Studio pipeline                         Voice Reader pipeline
record -> transcribe -> align -> dataset      import -> normalize -> segment
       -> train -> promote source -> convert/evaluate runtime
                                      |                   |
                                      +-> VoiceModelBundle -> loader -> synthesize
```

- Voice Studio owns data acquisition, training history, evaluation, and bundle
  promotion. Deleting Studio must not invalidate an already promoted bundle.
- Voice Reader owns document preparation, model selection, synthesis jobs,
  chunk caching, and export. Deleting Reader must not affect Studio artifacts.
- The two apps do not import one another. Their only production seam is a
  directory conforming to Bundle Contract v1.
- Shared contract code contains schemas, typed errors, and provider-neutral
  ports only. It contains no UI, training workflow, F5 runtime, or app state.
- Qwen3-TTS 12Hz 0.6B Base is the leading candidate, not an unconditional
  backend decision. Its reference-audio path is a zero-shot baseline until a
  fine-tuned artifact can make the complete Studio-to-Reader round trip.
  Official remote CUDA fine-tuning produces the source checkpoint; community
  MLX-Audio provides a separately verified local conversion/inference runtime.
  Community MLX support is never described as official Qwen support.
- F5-TTS remains a private-use comparison backend behind the same ports. Its
  official code provenance (MIT), pretrained-weight provenance (CC BY-NC), and
  non-official community MLX provenance are preserved in every F5 artifact.
  Adding or removing F5 does not change the bundle envelope or Reader.
- The immediate product is strictly local, personal, and non-commercial. The
  provider-neutral license schema still records each selected artifact's actual
  code, weight, derivative, runtime, and usage terms separately. This document
  does not grant broader rights.

## Independently deletable modules

Each module has one job and one public seam. Composition roots wire modules;
they do not absorb domain behavior.

| Module | One job | Input | Output / public seam | Owner | Failure boundary |
|---|---|---|---|---|---|
| `script_manifest_translation` | Translate a reviewed upstream script JSONL into the lifecycle schema | Script-manifest path plus declared schema id/version | `RecordingPromptManifest` | Studio | Reject unknown versions or malformed rows; never mutate the source |
| `recording_capture` | Persist lossless session masters and take metadata | `RecordingPromptManifest`, audio device samples | `RecordingSessionManifest` plus WAV masters | Studio | An incomplete session remains resumable and cannot enter transcription |
| `transcription_adapter` | Transcribe one recording session through the configured MLX implementation | Accepted session manifest and masters | `TranscriptionManifest` | Studio | Runtime/model failure preserves recording inputs and writes no accepted manifest |
| `script_aligner` | Align transcript spans to expected utterances | Prompt, session, and transcription manifests | `AlignmentManifest` | Studio | Uncertain/mismatched rows become `needs_review`; they are never silently accepted |
| `audio_preprocessor` | Decode, trim, resample, and segment audio deterministically | Human-accepted alignment rows and source WAVs | Preprocessed WAV clips plus preprocessing report | Studio | Invalid audio/provenance rows are rejected; the CPU utility owns no dataset policy |
| `dataset_builder` | Index clips and create session-exclusive splits | Accepted alignment rows and preprocessing report | `DatasetManifest` | Studio | Invalid text/provenance rows are rejected with reasons, not partially admitted |
| `remote_training_run` | Execute one official backend fine-tuning recipe resumably on explicit remote CUDA | Dataset manifest, base-model provenance, training config, remote target | `TrainingRunManifest` plus source checkpoints | Studio | Interruption leaves a valid latest checkpoint; no local/CPU training fallback occurs |
| `source_model_evaluator` | Score one source checkpoint against the frozen acceptance set | Source checkpoint, test manifest, evaluation config | `SourceEvaluationReport` and preview audio | Studio | A failed threshold blocks source promotion without deleting the checkpoint |
| `source_model_promoter` | Freeze one accepted checkpoint and evidence as a model version | Accepted source evaluation, checkpoint, metadata, licenses | Immutable `SourceModelRelease` | Studio | Existing model versions are never overwritten; no runtime conversion is implied |
| `runtime_variant_converter` | Convert one promoted source release into one local runtime format | Source release plus converter contract | `ConversionManifest` plus immutable runtime candidate | Studio | Source release remains intact; failed conversion creates no selectable variant |
| `runtime_variant_evaluator` | Compare one runtime candidate with its source release | Runtime candidate, source release, frozen parity prompts | `RuntimeEvaluationReport` and preview audio | Studio | Failed parity blocks variant selection without revoking the source release |
| `bundle_publisher` | Package one source release and accepted variants into one immutable bundle | Source release, accepted runtime reports, metadata, licenses, references | `VoiceModelBundle` plus atomic promotion pointer | Studio | Existing bundles are never overwritten; publication is atomic or absent |
| `bundle_reader` | Parse and verify a bundle without loading a model | Bundle directory | `VerifiedVoiceModelBundle` | Shared | Typed schema, path, checksum, or provenance error occurs before backend initialization |
| `voice_model_catalog` | List installed immutable bundles for explicit user selection | Voice-model roots and verified bundle metadata | `VoiceModelDescriptor[]` | Reader | It never chooses a voice implicitly or loads model weights |
| `local_capability_detector` | Report local execution capabilities | OS/runtime probes | `LocalCapabilitySet` | Shared | Probe errors are data in the report; they do not invent GPU support |
| `remote_training_target` | Verify one explicitly configured remote CUDA target | Endpoint/config reference | `RemoteTrainingCapability` | Studio | Unavailable or incompatible CUDA fails before data export or training |
| `capability_selector` | Choose a supported device under caller policy | Capability set, backend support, `ExecutionPolicy` | `ExecutionPlan` | Shared | No allowed compatible device produces `NoCompatibleDevice` |
| `voice_model_loader` | Load a verified bundle as an opaque model handle | Verified bundle, execution plan, backend registry | `VoiceModelLoading.load` -> `LoadedVoiceModel` | Reader-facing shared port | Unknown backend/version or load failure is typed and never falls through to another model |
| `inference_adapter` | Synthesize one chunk with a loaded handle | `SynthesisRequest`, loaded handle | `SpeechSynthesizing.synthesize` -> WAV `AudioChunk` | Reader-facing shared port | Failed chunks remain retryable; prior cached chunks are untouched |
| `qwen3_mlx_converter` | Convert one promoted Qwen source release with pinned MLX-Audio | Source release plus conversion request | Qwen `RuntimeVariantCandidate` | Backend | Non-official converter provenance is explicit; it never loads Reader models |
| `qwen3_mlx_loader` | Load one verified Qwen MLX runtime variant | Verified variant plus execution plan | Opaque Qwen handle through `VoiceModelLoading` | Backend | It never converts, downloads, selects, or synthesizes |
| `qwen3_mlx_synthesizer` | Synthesize one chunk with a loaded Qwen handle | Synthesis request plus opaque handle | WAV `AudioChunk` through `SpeechSynthesizing` | Backend | It owns no loading, caching, retry, or assembly |
| `qwen3_cuda_trainer` | Fine-tune Qwen3-TTS remotely with its official recipe | Explicitly exported dataset and training request | Resumable source checkpoints and training report | Backend | Network/remote/CUDA concerns end at this adapter; it never converts, promotes, or loads Reader models |
| `f5_mlx_converter` | Convert one promoted F5 comparison source with pinned community code | Source release plus conversion request | F5 `RuntimeVariantCandidate` | Backend | Private-use provenance stays isolated; it never loads Reader models |
| `f5_mlx_loader` | Load one verified F5 MLX comparison variant | Verified variant plus execution plan | Opaque F5 handle through `VoiceModelLoading` | Backend | It never converts, downloads, selects, or synthesizes |
| `f5_mlx_synthesizer` | Synthesize one comparison chunk with a loaded F5 handle | Synthesis request plus opaque handle | WAV `AudioChunk` through `SpeechSynthesizing` | Backend | It owns no loading, caching, retry, or assembly |
| `f5_cuda_trainer` | Fine-tune F5 comparison models remotely with its official recipe | Explicitly exported dataset and training request | Resumable source checkpoints and training report | Backend | Private-use derivative terms are recorded; it never converts, promotes, or loads Reader models |

Deletion tests are architectural tests: shared contracts use a fake backend;
Reader tests use a fixture bundle without Studio; Studio tests promote without
Reader; Qwen and F5 adapter tests run outside shared-contract tests.

## ICM product lifecycle

The lifecycle is a pipeline whose status is visible from files. A run is a new
copy of a declared run template; no stage searches the whole repository.

```text
data/voices/<voice-id>/runs/<run-id>/
  01_recording/       CONTEXT.md, input/, output/recording-session.json
  02_transcription/   CONTEXT.md, output/transcription.jsonl
  03_alignment/       CONTEXT.md, output/alignment.jsonl
  04_dataset/         CONTEXT.md, output/dataset.json, output/audio/<style>/*.wav
  05_training/        CONTEXT.md, output/training-run.json, output/checkpoints/
  06_source-evaluation/ CONTEXT.md, output/source-evaluation.json, output/previews/
  07_source-promotion/  CONTEXT.md, output/source-release.json
  08_runtime-conversion/ CONTEXT.md, output/conversion.json, output/candidate/
  09_runtime-evaluation/ CONTEXT.md, output/runtime-evaluation.json, output/previews/
  10_bundle-publication/ CONTEXT.md, output/publication.json

data/voices/<voice-id>/models/
  versions/<model-version>/     immutable VoiceModelBundle
  promoted.json                 atomic pointer to one version
```

The numbered run stages are working products. Backend rules, JSON schemas, base-
model provenance records, and blank run templates are stable factory
material and must live outside run outputs. Every JSON/JSONL manifest is a
plain, human-editable gate. Binary audio and weights are referenced by relative
path plus checksum; they are never embedded in a manifest.

The existing `PLAN.md` path `data/dataset/<style>/audio/` is a conceptual seam.
Its versioned realization is the dataset stage's
`output/audio/<style>/*.wav`, preventing one run from overwriting another.

## Working artifact contracts

All schema versions are SemVer strings. IDs are opaque, stable strings. All
paths inside a manifest are normalized relative paths using `/`; absolute paths
are forbidden in portable artifacts.

### `RecordingPromptManifest` v1

One JSON document identifies source manifest schema/version and ordered
utterances. Each utterance carries `utterance_id`, exact `text`, canonical
`style`, source provenance, and session/block intent. Translation is the only
place aware of the current Claude worktree's uncommitted bus/schema-v1 shape.
Lifecycle modules never import `voiceclonegpt.bus.contracts`; a later adapter
may translate that envelope into this manifest.

### `RecordingSessionManifest` v1

One JSON document per session contains:

- `schema_version`, `session_id`, `voice_id`, `prompt_manifest_id`, timestamps,
  consent reference, recording conditions, and terminal status;
- lossless master WAV entries with relative path, SHA-256, sample rate, channel
  count, bit depth, duration, and device/setup provenance;
- take/marker observations and calibration/room-tone references.

Raw masters are append-only. Cleanup, resampling, and splitting create new
files; they never rewrite the originals.

### `TranscriptionManifest` v1

JSONL with one observed segment per row: `segment_id`, `session_id`, master
audio reference, start/end seconds, observed text, language, confidence, and
transcriber id/version/model checksum. The run header records local MLX Whisper
runtime provenance as described by `PLAN.md`. Transcription is evidence, not
ground truth, and does not overwrite expected script text.

### `AlignmentManifest` v1

JSONL with one expected utterance per row: `utterance_id`, transcript segment
ids, master audio reference, start/end seconds, expected text, observed text,
canonical style, confidence, mismatch reasons, and `review_state` (`pending`,
`accepted`, or `rejected`). Only a human can change `pending` to `accepted`.
Marker speech is represented as excluded evidence and never enters the dataset.

### `DatasetManifest` v1

One JSON document plus referenced WAV clips. Every admitted row contains
`utterance_id`, exact accepted text, canonical style, clip path/checksum/audio
properties, recording-session provenance, alignment reference, split
(`train`, `validation`, or `test`), and acceptance actor/time. Splits are
session-exclusive; the builder fails if a session appears in multiple splits.
The admission gate rejects the entire candidate clip when diarization marks
overlap or any non-target speaker intersects it; source separation cannot
promote a mixed-speaker clip. Rejected rows and reasons live in a separate
reviewable report.

### `TrainingRunManifest` v1

One JSON document records `run_id`, dataset id/checksum, backend id/version,
base-model source/revision/checksum/license, official remote-CUDA recipe,
hyperparameters, random seed, code/environment and remote-target provenance,
checkpoint index, resume lineage, metrics, timestamps, and status. Checkpoints
are append-only run products; they are never Reader inputs. Qwen and F5 training
use separate leaf adapters and neither has a local or CPU fallback in v1.

### `SourceEvaluationReport` v1

One JSON document binds a source checkpoint checksum to the frozen test-set
checksum, thresholds, measurements, source-runtime previews, comparison
baselines, and human decision. Source promotion requires `decision: accepted`,
all mandatory thresholds passing, and named/time-stamped human approval.

### `SourceModelRelease` v1

One immutable JSON document plus the copied source artifact records
`source_release_id`, `voice_id`, `model_version`, provider/model identity,
`artifact_kind` (`reference_clone`, `fine_tuned_full`, or
`fine_tuned_adapter`), checkpoint or adapter format/revision/checksum, base
model dependency when the artifact is an adapter, dataset/training/evaluation
checksums, license layers, reference-library id, timestamps, and approval. A
`reference_clone` has no learned personal checkpoint and is baseline-only; a
fine-tuned release is the canonical promoted learned model. It contains no MLX
conversion and is never rewritten when runtime variants are added or removed.

### `ConversionManifest` v1

One JSON document binds an immutable promoted `SourceModelRelease` checksum to
an immutable local runtime-candidate checksum. It records converter
id/source/revision, official or community status, source/target artifact-format
versions, MLX/runtime versions, quantization, tensor/dtype mapping, validation
results, warnings, and timestamps. Conversion must be reproducible from the
source release and must not mutate or supersede it.

### `RuntimeEvaluationReport` v1

One JSON document binds a runtime-candidate checksum to its source-release
checksum and frozen parity-set checksum. It records intelligibility, alignment,
duration, performance, generated previews, differences from source-runtime
output, and human decision. A variant is selectable only when
`decision: accepted`, mandatory parity thresholds pass, and approval identifies
the exact source and runtime checksums.

The provider-neutral `runtime_parity` seam first calls
`evaluate(RuntimeParityRequest, RuntimeParityEvaluator)`. The request binds the
exact source-release, runtime-candidate, and parity-set SHA-256 values plus each
ordered prompt, source-audio checksum, and mandatory metric threshold. The
evaluator returns candidate-audio checksums and measurements; even when every
metric passes, the resulting report is always `pending_listening`.

Only `approve(RuntimeEvaluationReport, ListeningApproval)` can produce an
accepted report. Approval requires a named listener, a time-zone-aware
timestamp, every ordered prompt id, and exact source, candidate, parity-set,
and report checksums. Missing or failed metrics, partial listening, or changed
evidence fails closed. This seam has no runtime registration or bundle
publication authority; those remain later, separately wired stages.

## Bundle Contract v1

A model version is an immutable directory. Partial bundles are built in a
sibling staging directory, verified, then atomically renamed.

```text
<bundle-id>/
  bundle.json
  checksums.sha256
  source/
    source-release.json
    checkpoint/             provider-owned promoted source payload
  runtimes/
    <runtime-variant-id>/
      variant.json
      model/                runtime-owned immutable payload
  references/
    references.json
    *.wav
  licenses/
    usage.json
    notices/                selected provider/runtime notices
  evaluation/
    source-report.json
    runtime-reports/*.json
    previews/*.wav
```

`bundle.json` contains these required provider-neutral fields:

| Field | Meaning |
|---|---|
| `bundle_schema_version` | Bundle envelope SemVer; starts at `1.0.0` |
| `bundle_id` | `<voice-id>@<model-version>` |
| `voice_id`, `model_version` | Stable voice identity and immutable promoted SemVer |
| `created_at`, `promoted_at` | UTC timestamps |
| `source_model` | Provider/model id, `artifact_kind`, source-release path/checksum, checkpoint or adapter format/version, base dependency when applicable, license references |
| `runtime_variants` | Zero or more variant ids with backend/runtime id, format, converter, quantization, artifact paths/checksums, parity-report path, and license references |
| `references` | Path to reference index, default reference id, style coverage |
| `training_provenance` | Dataset/training/evaluation ids and SHA-256 values; no raw private transcript content |
| `compatibility` | Minimum loader contract version and supported synthesis features |
| `audio_contract` | Output sample rate/channel/encoding guarantees or declared ranges |
| `license` | Path to usage record plus separate code, source-weight, derivative, converter/runtime, and output-use notices |
| `integrity` | Algorithm (`sha256`) and checksum-manifest path |

`checksums.sha256` covers every file except itself and uses sorted relative
paths. No symlink may resolve outside the bundle. Unknown optional fields are
preserved by tools; unknown required major schema versions fail closed.

The source checkpoint and every runtime variant are distinct immutable
artifacts. A runtime variant references exactly one source-release checksum and
may not claim a model version of its own. Removing a variant leaves the source
release valid; adding a variant publishes a new immutable bundle package rather
than mutating an existing directory.

Reference clips are curated, consented WAV files included by value, not paths
back into a training run. `references.json` records id, relative path/checksum,
exact transcript, style, audio properties, duration, provenance session, and
whether the clip is the default. At least one verified default clip is required.

### Version and promotion rules

- `bundle_schema_version` versions the envelope. A loader accepts a supported
  major and any minor whose required-feature set it understands.
- `model_version` versions the promoted source model's voice behavior. Patch =
  retraining or metadata/package correction under the same schema; minor = a
  compatible added style/capability; major = an incompatible source tensor or
  loader-required contract. Runtime conversion or quantization does not create
  a new model version.
- `bundle_revision` versions packaging of one source release. Adding/removing a
  runtime variant increments this immutable package revision without changing
  `model_version`; Reader resolves a concrete bundle checksum, never “latest.”
- `promoted.json` contains only `voice_id`, `model_version`, `bundle_id`, bundle
  relative path, bundle checksum, and update timestamp. Update uses atomic file
  replacement after the destination bundle verifies.
- Source promotion copies the accepted checkpoint out of `05_training` before
  conversion. Bundle publication never points Reader into a run and never
  deletes checkpoints/releases. Rollback changes only `promoted.json` to
  another already verified immutable bundle revision.

### Provider/runtime provenance and isolation

The first source provider is Qwen3-TTS 12Hz 0.6B Base. Its official remote-CUDA
fine-tuning output is copied unchanged into `source/checkpoint/`; a community
MLX-Audio conversion is a child runtime variant under `runtimes/`. Metadata must
say `upstream_status: community`, identify the MLX-Audio repository, revision,
converter configuration, and checksum, and must not call that runtime official
Qwen MLX support.

F5-TTS is an independently deletable private-use comparison provider. An F5
bundle follows the identical source/variant structure, but records official F5
code provenance (MIT), official pretrained weights and derivative provenance
(CC BY-NC), and the non-official `f5-tts-mlx` runtime separately. Product copy
may say “F5-compatible MLX runtime,” not “official F5 MLX support.”

Provider weights, download caches, tokenizer/vocabulary assets, remote CUDA
checkpoints, and MLX candidates remain physically confined to the selected
provider cache, numbered run stage, source release, or runtime-variant folder.
Shared packages contain no Qwen, F5, MLX, PyTorch, or CUDA imports or assets.

`licenses/usage.json` records:

- intended local/private usage and a human-readable usage note;
- provider code source/revision/license;
- pretrained-weight source URL/repository revision/checksum/license;
- fine-tuned derivative terms distinct from source-weight terms;
- converter/runtime source/revision/license and official/community status;
- the fine-tune's derivative lineage and dataset/voice consent references;
- output-use terms, if any, separately from code and artifact terms;
- metadata author, recorded-at timestamp, and notice-file paths.

Missing or contradictory provenance is `InvalidLicenseMetadata` and blocks
promotion and loading. These are engineering safeguards and provenance records,
not a legal determination. Qwen's selected official code/weights are recorded
as Apache-2.0; F5's selected official weights and their fine-tunes retain the
CC-BY-NC/personal-noncommercial restriction. Any future provider gets a leaf
adapter and its own provenance; the bundle envelope stays neutral.

### Producer/consumer model handoff

The two apps do not need to use the same model family, and the Reader does not
assume that the model used for training is the model used for local generation.
The rules are explicit:

| Concern | Voice Studio (producer) | Voice Reader (consumer) |
|---|---|---|
| Voice identity | Creates a stable `voice_id` and immutable model versions | Selects any installed `voice_id` and exact bundle checksum, including another consented voice |
| Learned artifact | Emits a full fine-tuned checkpoint or a fine-tuned adapter with its base dependency | Loads that artifact through an accepted runtime variant; never trains, edits, or silently substitutes it |
| Model family | Qwen, F5, or another provider behind a leaf trainer | Any provider with a verified loader/runtime adapter |
| Cross-family use | Requires a declared converter/export and parity evidence | Fails with a typed compatibility error when no verified path exists |
| Reference audio | Curated evidence and optional zero-shot baseline | Optional conditioning only when the selected runtime declares it |

`reference_clone` means “base model plus reference audio”; it is not a trained
personal model. `fine_tuned_full` means complete learned weights.
`fine_tuned_adapter` means adapter weights plus the immutable base-model
dependency. Only the latter two kinds can satisfy the trained-model acceptance
gate; reference cloning remains a comparison mode.

Reader receives one explicit bundle for a synthesis job, or a descriptor from
`voice_model_catalog` and then an explicit bundle selection. It never searches
training runs, chooses “latest,” or assumes that a Qwen artifact can be loaded
by an F5 runtime merely because both produce speech.

## Stable Reader interfaces

The public interfaces are provider-neutral and live below both apps. Names are
normative; language-specific syntax is illustrative.

```text
VoiceModelLoading.load(ModelLoadRequest) async
  -> LoadedVoiceModel
  throws VoiceModelLoadError

ModelLoadRequest = {
  bundle_url,
  execution_policy,
  runtime_preference,      // explicit variant id or provider-neutral `auto`
  supported_bundle_schema_major: 1
}

LoadedVoiceModel = {
  descriptor,             // source release + selected runtime metadata
  execution_plan,         // actual runtime/device/dtype selected
  opaque_backend_handle   // not serialized or inspected by Reader
}

SpeechSynthesizing.synthesize(SynthesisRequest, LoadedVoiceModel) async
  -> AudioChunk
  throws SynthesisError
```

`VoiceModelLoading` has one job: produce a verified, ready opaque handle. Its
implementation delegates path/schema/integrity work to `bundle_reader`, device
choice to `capability_selector`, and provider loading to the registered backend.
It selects only a checksum-verified runtime variant whose parity report passed.
It does not convert, synthesize, cache, promote, discover UI state, or download
weights. Absence of an accepted compatible variant is a typed failure.

`SpeechSynthesizing` has one job: synthesize one normalized chunk. A request
contains text, canonical style, reference id, deterministic seed where
supported, and generation settings. It never reads arbitrary filesystem paths.
Voice Reader owns chunk retries/caching/assembly outside the adapter.

Required loader errors are `BundleNotFound`, `UnsupportedBundleSchema`,
`MalformedManifest`, `UnsafeBundlePath`, `IntegrityMismatch`,
`InvalidLicenseMetadata`, `MissingReference`, `UnsupportedBackend`,
`RuntimeVariantUnavailable`, `RuntimeParityRejected`, `NoCompatibleDevice`, and
`BackendLoadFailed`. Errors include a safe user action and never expose private
transcript text.

## Local capability selection

`local_capability_detector` reports facts only: OS/architecture, CPU support,
local MLX GPU availability, memory when knowable, supported dtypes, and backend
runtime versions. A provider runtime adapter separately reports which detected
combinations it actually supports. `capability_selector` intersects the two
with this policy:

1. An explicitly requested compatible device wins or fails; it never silently
   changes device.
2. For local work, `auto` prefers a compatible MLX GPU, then MLX CPU when both
   caller policy and backend support allow it. Local CUDA/PyTorch is not an
   implicit fallback.
3. Qwen generation uses MLX GPU when available. MLX CPU fallback occurs only
   when the request policy allows it and the Qwen MLX probe reports support; the
   returned execution plan records the downgrade for UI disclosure.
4. `gpu_required` generation fails when no supported MLX GPU exists. Backend
   load or out-of-memory failure does not silently retry CPU; Reader may offer
   an explicit new request with MLX CPU allowed.

Remote training does not use the local selector. `remote_training_target`
verifies an explicitly configured CUDA target and official provider recipe
before any export. Qwen and F5 fine-tuning are remote-CUDA-only in v1 because no
reliable MLX recipe is established; an unavailable target fails before training
and never falls back to CPU, local PyTorch, or another provider. A future MLX
trainer is a new leaf capability, not an automatic route change.

Training and synthesis persist the actual target/device, dtype, runtime, and
fallback decision in their respective manifests. Capability modules never
download, load, train, convert, or synthesize a model.

## MLX-first compute boundaries

MLX is the preferred tensor/runtime layer for every local stage it can reliably
support. “Preferred” is enforced by adapter selection and recorded execution
plans, not by importing MLX into domain contracts.

| Stage | Preferred implementation | Allowed non-MLX boundary and reason |
|---|---|---|
| Transcription | Configurable Apple MLX Whisper adapter from `PLAN.md` | WAV/container I/O returns PCM and owns no transcript policy |
| Alignment | Validated MLX-Audio forced aligner | Script-aware deterministic CPU alignment is an explicit fallback behind the same port |
| Decode/resample/segment | Ordinary CPU tools such as ffmpeg/soundfile-compatible helpers | This is the preferred deterministic implementation; MLX adds no useful acceleration |
| Qwen fine-tuning | None established locally in v1 | Isolated official Qwen remote PyTorch/CUDA adapter; no other stage imports it |
| F5 fine-tuning | None established locally in v1 | Isolated official F5 remote PyTorch/CUDA adapter for private comparison only |
| Model conversion | Pinned MLX-Audio conversion of a promoted Qwen source release | Reading a PyTorch checkpoint is confined to converter input; emitted candidate remains a runtime variant |
| Qwen inference/generation | Pinned community MLX-Audio Qwen3-TTS 0.6B runtime | No implicit PyTorch fallback; incompatibility is a typed load/capability failure |
| F5 inference/generation | Pinned community `f5-tts-mlx` comparison runtime | Official PyTorch/MPS may serve only as parity reference, not a silent Reader fallback |
| Evaluation | Local MLX generation plus MLX Whisper/alignment metrics and human listening | Deterministic file checks/scalar statistics use ordinary CPU code; remote metrics are out of scope |
| Local serving | Loopback-only MLX-Audio service or in-process MLX adapter | Swift/XPC or HTTP transport, authentication, and file streaming own no model logic |

Every non-MLX component is a leaf behind the seam named above. Its manifest
records implementation, version, input/output checksums, and why it was used.
Adding a non-MLX fallback to a local model-compute stage is an architecture
change, not a runtime convenience.

## Acceptance contract

Coding may advance past scaffolding only when automated fixtures cover these
contracts; hardware-dependent F5 tests may be separately tagged but must have a
deterministic fake-backend contract suite.

1. A valid v1 F5-compatible MLX fixture bundle verifies and loads through
   `VoiceModelLoading`; one short synthesis produces a WAV matching its declared
   audio contract.
2. Reader loads an immutable fixture with Studio absent. Studio promotes an
   accepted fake-backend candidate with Reader absent. Shared contracts test
   with the F5 package absent.
3. Corrupting any payload byte yields `IntegrityMismatch` before backend load;
   path traversal/symlink escape yields `UnsafeBundlePath`.
4. Unsupported major schemas, missing artifacts/references, unknown backends,
   and incomplete F5 license/provenance each produce their typed failure.
5. Promotion refuses a failed/unapproved evaluation, never overwrites an
   existing version, publishes no partial bundle after an injected failure, and
   atomically updates `promoted.json` only after verification.
6. A promoted bundle contains no pointer into a run, no unlisted CUDA/PyTorch
   checkpoint, and no absolute path. Reader never scans `05_training` or
   chooses/converts a checkpoint itself.
7. Dataset validation proves exact accepted text, canonical styles, checksums,
   alignment provenance, and recording-session-exclusive train/validation/test
   splits. Pending/rejected alignment rows cannot enter training.
8. Capability fixtures prove: MLX GPU is preferred locally; F5 training selects
   a reliable MLX GPU trainer when registered or the explicit remote CUDA
   trainer otherwise; training with neither fails; generation with explicit
   MLX CPU allowance records CPU fallback; `gpu_required` never falls back.
9. F5 provenance tests require MIT code notice, CC BY-NC pretrained-weight
   notice, source revision/checksum lineage, personal/non-commercial scope, and
   consent references in the promoted bundle.
10. Restart tests resume training from the last indexed checkpoint and resume
    Reader generation at the first uncached chunk without altering accepted
    earlier artifacts.
11. Contract compatibility tests preserve unknown optional fields and reject an
    unknown required feature. Golden fixtures cover every supported bundle
    schema minor version.
12. Human acceptance records bind the listened evaluation previews to the exact
    checkpoint, dataset, and bundle checksums promoted.
13. MLX-boundary tests fail if local inference, conversion, serving, or learned
    evaluation imports the CUDA trainer; manifests identify every allowed
    non-MLX helper and community F5 MLX provenance is labeled non-official.
14. The model round-trip fixture trains or loads a `fine_tuned_full` or
    `fine_tuned_adapter` artifact in Studio, publishes a bundle, removes Studio
    and trainer imports, and proves Reader loads the same artifact/runtime and
    generates the fixed prompts with the expected voice and checksum.
15. A `reference_clone` fixture is clearly labeled baseline-only and cannot be
    promoted as the trained personal-model path.
16. A Reader catalog fixture lists at least two immutable voice bundles and
    proves the user can select either exact bundle without selecting “latest,”
    loading the wrong voice, or importing Studio.

## Non-goals and dependency rules

- This contract does not choose SwiftUI/AppKit structure, implement UI, or
  change the Claude script-ingestion worktree.
- App composition roots may depend on shared ports and inject adapters. Shared
  ports never depend on apps, F5, MLX, PyTorch, or CUDA. Backend adapters depend
  inward on shared ports only.
- UI may display progress and request commands; it does not align transcripts,
  build datasets, select checkpoints, validate bundles, or orchestrate retries.
- Event-bus messages may carry a typed manifest path and version, never model
  weights or audio inline. Bus schema v1 is translated at an app boundary and
  remains independent of lifecycle and bundle schema versions.
- There is no global lifecycle manager. Status is derived from each run's
  manifests and output folders; promotion state comes only from the immutable
  bundles and `promoted.json`.
