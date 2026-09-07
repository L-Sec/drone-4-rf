"""Drone 4-RF: passive SDR scanner for probable drone RF/EMI activity.

Receive-only by design. No module in this package imports or invokes any
transmit API.
"""

__version__ = "0.7.0"

LEGAL_NOTICE = (
    "Drone 4-RF is a PASSIVE, receive-only RF scanner. You are responsible\n"
    "for complying with local radio, aviation, surveillance, and privacy\n"
    "laws. Detections are probabilistic indicators, never proof."
)
