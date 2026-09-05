#!/usr/bin/env python3
"""Exact, rank-free effect coding for three semantic-opener contribution vectors."""

from __future__ import annotations


DELIMITER_TYPES = ("parenthesis", "square", "quote")


def centered_terms(terms):
    """Return the arithmetic shared term and zero-sum type contrasts.

    ``terms`` must contain the three same-context, token-aligned natural terms in
    DELIMITER_TYPES order. Values may be NumPy arrays or torch tensors.
    """
    if len(terms) != 3:
        raise ValueError("exactly three delimiter terms are required")
    shared = (terms[0] + terms[1] + terms[2]) / 3.0
    contrasts = tuple(term - shared for term in terms)
    return shared, contrasts


def replace_term(native_head_write, native_term, replacement_term):
    """Exact intervention at the final L13H8 projected-output write."""
    return native_head_write - native_term + replacement_term


def remove_contrast(native_head_write, native_term, shared_term):
    """Preserve the context-local shared term and delete only delimiter contrast."""
    return replace_term(native_head_write, native_term, shared_term)


def swap_contrast(native_head_write, native_term, shared_term, donor_contrast):
    """Preserve shared support and install another natural delimiter contrast."""
    return replace_term(native_head_write, native_term, shared_term + donor_contrast)


def remove_shared(native_head_write, shared_term):
    """Diagnostic complement: preserve native contrast while deleting shared support."""
    return native_head_write - shared_term


def closer_type_and_common_axes(logits, closer_indices, target_index):
    """Return centered target-closer evidence and common closer support.

    The common axis is capable of moving under an intervention; it is not a
    same-answer fixed-bar control.
    """
    if len(closer_indices) != 3 or target_index not in closer_indices:
        raise ValueError("three closer indices including target_index are required")
    common = sum(logits[index] for index in closer_indices) / 3.0
    return logits[target_index] - common, common
