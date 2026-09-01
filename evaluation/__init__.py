"""Evaluator-only package. This directory is outside the inference package."""

from evaluation.evaluator import EvaluatorHost
from evaluation.models import EvaluationArtifact

__all__ = ["EvaluationArtifact", "EvaluatorHost"]
