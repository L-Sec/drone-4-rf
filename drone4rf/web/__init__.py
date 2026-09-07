"""Browser dashboard (Stage 5b): stdlib HTTP server + Server-Sent Events.

No third-party dependencies and no build step - the whole UI is one
static HTML file served from this package. Binds to loopback by default;
see server.py for the local-service hardening notes.
"""

from drone4rf.web.server import run_server

__all__ = ["run_server"]
