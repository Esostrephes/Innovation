#!/usr/bin/env python3
# ============================================================
# run_server.py  —  UniAttend v3 launcher with pyngrok tunnel
# ============================================================
#
# Install:
#   pip install pyngrok uvicorn
#
# Usage:
#   python run_server.py
#
# With your ngrok authtoken (free at https://dashboard.ngrok.com):
#   NGROK_AUTHTOKEN=your_token python run_server.py
#
# Or set it once permanently:
#   ngrok config add-authtoken your_token
# ============================================================

import os
import time
import threading
import uvicorn
from pyngrok import ngrok, conf

# ── Optional: set authtoken from env var ─────────────────────
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "")
PORT = int(os.getenv("PORT", "8000"))

if NGROK_AUTHTOKEN:
    conf.get_default().auth_token = NGROK_AUTHTOKEN
    print(f"✓ ngrok authtoken set from environment")
else:
    print("⚠  No NGROK_AUTHTOKEN set — using anonymous tunnel (limited)")
    print("   Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken")
    print()

# ── Open ngrok tunnel ─────────────────────────────────────────
print(f"Opening ngrok tunnel on port {PORT}…")
tunnel = ngrok.connect(PORT, "http")
public_url = tunnel.public_url

# ngrok always gives http:// — upgrade to https
if public_url.startswith("http://"):
    public_url_https = public_url.replace("http://", "https://", 1)
else:
    public_url_https = public_url

print()
print("=" * 60)
print(f"  🌐  Public URL  :  {public_url_https}")
print(f"  🏠  Local URL   :  http://localhost:{PORT}")
print("=" * 60)
print()
print("Share the Public URL with students for mobile enrolment.")
print("Press Ctrl+C to stop.\n")

# ── Start uvicorn in a background thread ──────────────────────
def run_uvicorn():
    uvicorn.run(
        "attendance_system_v3:app",   # module:app
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )

server_thread = threading.Thread(target=run_uvicorn, daemon=True)
server_thread.start()

# ── Keep main thread alive; clean up on Ctrl+C ────────────────
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down…")
    ngrok.kill()
    print("Tunnel closed. Bye!")
