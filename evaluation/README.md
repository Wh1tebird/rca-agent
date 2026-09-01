# Evaluation boundary

Phase 2 implements a minimal evaluator-only post-freeze loader and separate Evaluation Artifact.
It runs only after a valid Answer Artifact is frozen and remains outside the investigation-policy
import graph. It proves lifecycle and isolation behavior; evaluator scoring is deferred.
