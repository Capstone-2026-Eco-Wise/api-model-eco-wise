"""
inference.py — Legacy wrapper for backward compatibility.
Imports the inference engine from the core module.
"""

from core.inference import (
    EcoWiseInference,
    ChannelAttentionLayer,
    FocalLoss,
    DEFAULT_CLASS_NAMES,
    IMG_SIZE,
)