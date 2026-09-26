"""Import-only zlib projection for WASI runtimes built without zlib.

urllib3 (pulled in eagerly by ``dspy.utils``) imports zlib at module load but
only uses it to decode HTTP bodies. The component never performs HTTP: network
actuation is owned by the host. Any actual compression call therefore traps
loudly instead of acquiring different semantics.
"""

from __future__ import annotations

MAX_WBITS = 15
DEFLATED = 8
Z_DEFAULT_COMPRESSION = -1
Z_SYNC_FLUSH = 2
Z_FINISH = 4


class error(Exception):
    pass


def _unsupported(*_args, **_kwargs):
    raise error("zlib is unavailable inside the dspy-wasm component")


compress = decompress = compressobj = decompressobj = crc32 = adler32 = _unsupported
