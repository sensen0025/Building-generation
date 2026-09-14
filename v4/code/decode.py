"""Strict, non-truncating decoder shared with the v5 source pipeline."""
from mcfix.containers import DecodeError, decode_path, varint_decode

def full_state_names(names):
    return list(names)
