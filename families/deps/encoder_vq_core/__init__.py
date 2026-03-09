"""Minimal dependency surface for encoder-side VQ models."""

from families.deps.encoder_vq_core.models_single_vq import (
    SingleVQWithEMA,
    TeacherStudentSingleVQ,
)

__all__ = ["SingleVQWithEMA", "TeacherStudentSingleVQ"]
