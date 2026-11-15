"""Run the Modal CLI with the CA bundle from SSL_CERT_FILE.

Modal's gRPC client verifies TLS against certifi's bundled list and ignores SSL_CERT_FILE,
so it fails behind a TLS-inspecting gateway whose CA is only in the configured bundle.
Verification stays on. Usage: python scripts/modal_ca.py <modal arguments>
"""

import os
import sys

import certifi

bundle = os.environ.get("SSL_CERT_FILE")
if bundle and os.path.isfile(bundle):
    certifi.where = lambda: bundle

from modal.__main__ import main  # noqa: E402

sys.argv[0] = "modal"
main()
