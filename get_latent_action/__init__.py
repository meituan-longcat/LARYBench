"""
Get Latent Action Module.

This module provides functionality for extracting latent actions from videos and images.
"""

from .dynamics import get_dynamic_tokenizer, get_latent_action

__all__ = ["get_dynamic_tokenizer", "get_latent_action"]
