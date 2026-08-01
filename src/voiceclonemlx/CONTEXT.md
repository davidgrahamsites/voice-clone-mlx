# VoiceCloneMLX domain package

One job: expose provider-neutral domain seams for Voice Studio and Voice
Reader. Individual modules own their own contracts; this package does not
coordinate pipelines or import UI/provider implementations.

## Inputs

Versioned manifests and small typed values defined by the named child module.

## Process

Keep domain decisions deterministic and dependency-light. Route model/runtime,
filesystem, and UI behavior through explicit adapters.

## Outputs

Small importable ports and value objects that can be tested without either app.

## Human check

Delete one child adapter and confirm the other domain seams and their tests
remain importable.
