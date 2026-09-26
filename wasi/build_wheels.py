#!/usr/bin/env python3
"""Cross-compile every native dependency of dspy-wasm for WebAssembly.

Target: componentize-py 0.25.1 components, i.e. CPython 3.14 on
``wasm32-wasip2``, extensions built as PIC shared libraries that the
componentize-py linker resolves against its embedded ``libpython3.14.so``.

Run ``wasi/toolchain.sh`` first; it writes ``build/wasi/env.sh``.

    python wasi/build_wheels.py fetch              # pinned sdists (wasi/native.txt)
    python wasi/build_wheels.py build [NAME ...]   # sdist -> wasi wheel (--missing: skip built)
    python wasi/build_wheels.py install            # wheels + pure deps -> build/wasi_deps

Recipes are data: Rust crates (maturin or setuptools-rust layouts) are built
with cargo exactly the way componentize-py builds its own PyO3 runtime; C,
Cython and mypyc packages go through their own PEP 517 backends with the
compiler and sysconfig pointed at the WASI CPython build.
"""

from __future__ import annotations

import base64
import csv
import email.parser
import hashlib
import io
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
SDISTS = BUILD / "sdists"
WORK = BUILD / "wasi_work"
WHEELS = BUILD / "wasi_wheels"
TARGET = BUILD / "wasi_deps"
WHEEL_TAG = "cp314-cp314-wasi_0_0_0_wasm32"


def toolchain() -> dict[str, str]:
    env_file = BUILD / "wasi" / "env.sh"
    if not env_file.exists():
        sys.exit("run wasi/toolchain.sh first")
    values = {}
    for line in env_file.read_text().splitlines():
        match = re.match(r'export (\w+)="(.*)"', line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


TC = toolchain()
SDK = Path(TC["WASI_SDK_PATH"])
DEPS = Path(TC["WASI_DEPS"])
PYBUILD = Path(TC["WASI_PYTHON_BUILD"])
SYSCONFIG_DIR = Path(TC["WASI_SYSCONFIG_DIR"])
HOST_PYTHON = TC["HOST_PYTHON"]
CLANG = f"{SDK}/bin/clang --target=wasm32-wasip2"
# wasi-libc keeps these POSIX surfaces behind opt-in emulation libraries;
# componentize-py ships all four as shared libraries in every component.
WASI_EMULATION_DEFINES = (
    "-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS "
    "-D_WASI_EMULATED_GETPID -D_WASI_EMULATED_MMAN"
)
# componentize-py's libpython3.14 is a shared library, while the reference
# build here is configured static; declare the shared ABI so PyAPI symbols
# keep default visibility under -fvisibility=hidden (numpy, ...).
PY_SHARED_ABI = "-DPy_ENABLE_SHARED=1"
WASI_EMULATION_LIBS = (
    "-lwasi-emulated-signal -lwasi-emulated-process-clocks "
    "-lwasi-emulated-getpid -lwasi-emulated-mman"
)


def _sysconfig() -> dict:
    module = next(SYSCONFIG_DIR.glob("_sysconfigdata_*.py"))
    return runpy.run_path(str(module))["build_time_vars"] | {"__module__": module.stem}


SYSCONFIG = _sysconfig()
EXT_SUFFIX = SYSCONFIG["EXT_SUFFIX"]


# ------------------------------------------------------------------ recipes


@dataclass
class CratePatch:
    """A crates.io dependency ported to WASI: a patched copy wired in via
    ``[patch.crates-io]`` in the recipe's workspace manifest."""

    name: str
    version: str
    edits: tuple[tuple[str, str, str], ...] = ()
    files: dict[str, str] = field(default_factory=dict)
    replace_all: tuple[tuple[str, str, str], ...] = ()  # (glob, old, new) across files


RUSTLS_NATIVE_CERTS_WASI = CratePatch(
    "rustls-native-certs",
    "0.8.4",
    edits=(
        (
            "src/lib.rs",
            '#[cfg(target_os = "macos")]\nuse macos as platform;\n',
            '#[cfg(target_os = "macos")]\nuse macos as platform;\n\n'
            '#[cfg(target_os = "wasi")]\nmod wasi;\n'
            '#[cfg(target_os = "wasi")]\nuse wasi as platform;\n',
        ),
    ),
    files={
        "src/wasi.rs": (
            "//! WASI has no system trust store. SSL_CERT_FILE / SSL_CERT_DIR are\n"
            "//! honoured by `load_native_certs` before this fallback is consulted;\n"
            "//! without them there are no platform roots to report.\n"
            "use crate::CertificateResult;\n\n"
            "pub fn load_native_certs() -> CertificateResult {\n"
            "    CertificateResult::default()\n"
            "}\n"
        )
    },
)


_UNIX = '#[cfg(target_family = "unix")]'
_UNIX_OR_WASI = '#[cfg(any(target_family = "unix", target_os = "wasi"))]'
GCP_AUTH_WASI = CratePatch(
    "gcp_auth",
    "0.12.7",
    # WASI is POSIX-shaped: $HOME/.config credentials; `gcloud` resolves, and
    # spawning it fails at runtime with std's "unsupported" (no processes).
    edits=tuple(
        (path, f"{_UNIX}\n{item}", f"{_UNIX_OR_WASI}\n{item}")
        for path, item in (
            ("src/config_default_credentials.rs", "use std::env;"),
            ("src/config_default_credentials.rs", "fn config_dir()"),
            ("src/config_default_credentials.rs", "const CONFIG_DIR"),
            ("src/gcloud_authorized_user.rs", "const GCLOUD_CMD"),
        )
    ),
)


AZURE_IDENTITY_WASI = CratePatch(
    "azure_identity",
    "1.0.0",
    # tokio has no process module on WASI (no processes exist); the Azure CLI
    # credential reports Unsupported and the credential chain moves on.
    edits=(
        (
            "src/process/tokio.rs",
            "        ::tokio::process::Command::new(program)\n"
            "            .args(args)\n"
            "            .output()\n"
            "            .await\n",
            '        #[cfg(target_os = "wasi")]\n'
            "        {\n"
            "            let _ = (program, args);\n"
            '            return Err(io::Error::new(io::ErrorKind::Unsupported, "WASI has no processes"));\n'
            "        }\n"
            '        #[cfg(not(target_os = "wasi"))]\n'
            "        ::tokio::process::Command::new(program)\n"
            "            .args(args)\n"
            "            .output()\n"
            "            .await\n",
        ),
    ),
)


# Rust's std panics in process::id() on WASI ("no pids on this platform").
# A component is a single process that cannot fork: a fixed identity is exact.
WASI_PID = (
    "**/*.rs",
    "std::process::id()",
    '(if cfg!(target_os = "wasi") { 1u32 } else { std::process::id() })',
)

_BROWSER_WASM = 'all(target_arch = "wasm32", not(target_os = "wasi"))'


def browser_wasm_only(name: str, version: str) -> CratePatch:
    """Crates whose `wasm32` code paths mean "browser" (wasm-bindgen, fetch).

    WASI is not a browser: narrow those gates so WASI takes the native paths.
    (reqwest >= 0.13.5 already makes this distinction upstream.)
    """
    return CratePatch(
        name,
        version,
        replace_all=tuple(
            (pattern, 'target_arch = "wasm32"', _BROWSER_WASM)
            for pattern in ("src/**/*.rs", "Cargo.toml")
        ),
    )


REQWEST_WASI = browser_wasm_only("reqwest", "0.12.28")


OS_STR_BYTES_WASI = CratePatch(
    "os_str_bytes",
    "6.6.1",
    # std::os::wasi is still feature-gated on wasip2. WASI OS strings are
    # arbitrary bytes (as on unix), so std's stable encoded-bytes API is exact.
    edits=(
        (
            "src/common/mod.rs",
            '#[cfg(target_os = "wasi")]\nuse std::os::wasi as os;\n',
            '#[cfg(target_os = "wasi")]\nmod os {\n'
            "    pub(super) mod ffi {\n"
            "        use std::ffi::{OsStr, OsString};\n"
            "        pub(crate) trait OsStrExt {\n"
            "            fn from_bytes(slice: &[u8]) -> &Self;\n"
            "            fn as_bytes(&self) -> &[u8];\n"
            "        }\n"
            "        impl OsStrExt for OsStr {\n"
            "            fn from_bytes(slice: &[u8]) -> &Self {\n"
            "                // SAFETY: every byte sequence is a valid OS string on WASI.\n"
            "                unsafe { OsStr::from_encoded_bytes_unchecked(slice) }\n"
            "            }\n"
            "            fn as_bytes(&self) -> &[u8] {\n"
            "                self.as_encoded_bytes()\n"
            "            }\n"
            "        }\n"
            "        pub(crate) trait OsStringExt {\n"
            "            fn from_vec(vec: Vec<u8>) -> Self;\n"
            "            fn into_vec(self) -> Vec<u8>;\n"
            "        }\n"
            "        impl OsStringExt for OsString {\n"
            "            fn from_vec(vec: Vec<u8>) -> Self {\n"
            "                // SAFETY: every byte sequence is a valid OS string on WASI.\n"
            "                unsafe { OsString::from_encoded_bytes_unchecked(vec) }\n"
            "            }\n"
            "            fn into_vec(self) -> Vec<u8> {\n"
            "                self.into_encoded_bytes()\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n",
        ),
    ),
)


AWS_LC_SYS_WASI = CratePatch(
    "aws-lc-sys",
    "0.39.0",
    edits=(
        # WASI has TCP/UDP sockets but no Unix-domain sockets.
        (
            "aws-lc/crypto/bio/internal.h",
            "#if defined(AF_UNIX) && !defined(OPENSSL_WINDOWS) && !defined(OPENSSL_ANDROID)\n",
            "#if defined(AF_UNIX) && !defined(OPENSSL_WINDOWS) && !defined(OPENSSL_ANDROID) && \\\n"
            "    !defined(__wasi__)\n",
        ),
        # No terminal on WASI: use the port in console_wasi.c.
        (
            "aws-lc/crypto/console/console.c",
            "#include <assert.h>\n",
            '#if defined(__wasi__)\n#include "console_wasi.c"\n#else\n#include <assert.h>\n',
        ),
        (
            "aws-lc/crypto/console/console.c",
            "    return ok;\n}\n",
            "    return ok;\n}\n#endif  // __wasi__\n",
        ),
    ),
    files={
        "aws-lc/crypto/console/console_wasi.c": (
            ROOT / "wasi" / "aws_lc_console_wasi.c"
        ).read_text()
    },
)


@dataclass
class Rust:
    """A PyO3 crate. ``module`` is the dotted import path of the extension."""

    sdist: str
    module: str
    manifest: str = "Cargo.toml"
    python_source: str | None = None  # dir holding python packages; None = synthesize
    python_packages: tuple[str, ...] = ()  # top-level packages to copy from python_source
    features: tuple[str, ...] = ()
    no_default_features: bool = False
    env: dict[str, str] = field(default_factory=dict)
    patches: tuple[tuple[str, str, str], ...] = ()
    rustflags: str = ""
    crate_patches: tuple[CratePatch, ...] = ()
    workspace: str | None = None  # manifest holding [patch.crates-io]; default: manifest
    replace_all: tuple[tuple[str, str, str], ...] = ()  # (glob, old, new) in the sdist
    cargo_update: tuple[str, ...] = ()  # semver-compatible lockfile bumps


@dataclass
class Pep517:
    """A C/Cython/mypyc package built by its own backend, cross-configured."""

    sdist: str
    env: dict[str, str] = field(default_factory=dict)
    config_settings: tuple[str, ...] = ()
    # (path, old, new): source edits applied once after unpacking.
    patches: tuple[tuple[str, str, str], ...] = ()


@dataclass
class Meson(Pep517):
    """A meson-python package; a WASI cross file is generated and passed in."""

    setup_args: tuple[str, ...] = ()


@dataclass
class Stdlib:
    """CPython stdlib extension modules componentize-py's runtime omits.

    Built from the matching CPython source exactly as on Linux, where they
    are shared extensions against libpython (only its exported API).
    """

    name: str
    version: str
    modules: dict[str, str]  # extension name -> Modules/ source
    libs: tuple[str, ...] = ()  # static archives from the WASI deps prefix


RECIPES: dict[str, Rust | Pep517 | Stdlib] = {
    # --- Rust / PyO3 -------------------------------------------------------
    "pydantic-core": Rust(
        "pydantic_core-2.46.5",
        "pydantic_core._pydantic_core",
        python_source="python",
        python_packages=("pydantic_core",),
    ),
    "rpds-py": Rust("rpds_py-2026.6.3", "rpds.rpds"),
    "jiter": Rust("jiter-0.17.0", "jiter.jiter", manifest="crates/jiter-python/Cargo.toml"),
    "orjson": Rust(
        "orjson-3.12.0", "orjson.orjson", python_source="pysrc", python_packages=("orjson",)
    ),
    "tiktoken": Rust(
        "tiktoken-0.14.0",
        "tiktoken._tiktoken",
        python_source=".",
        python_packages=("tiktoken", "tiktoken_ext"),
        features=("python",),  # as setup.py
    ),
    "tokenizers": Rust(
        "tokenizers-0.23.2",
        "tokenizers.tokenizers",
        manifest="bindings/python/Cargo.toml",
        python_source="py_src",
        python_packages=("tokenizers",),
        # esaxx is C++; rustc links via clang, so name the C++ runtime (the
        # component bundles libc++/libc++abi).
        rustflags="-Clink-args=-lc++ -Clink-args=-lc++abi",
    ),
    "fastuuid": Rust(
        "fastuuid-0.14.0",
        "fastuuid.fastuuid",
        # pyo3 0.26 reaches the still-unstable std::os::wasi on wasip2;
        # pyo3 0.28 (what componentize-py's runtime uses) handles WASI
        # strings with stable APIs.
        patches=(("Cargo.toml", 'version = "0.26"', 'version = "0.28"'),),
        rustflags="-Adeprecated",  # the crate denies warnings; 0.28 deprecates APIs it uses
    ),
    "hf-xet": Rust(
        "hf_xet-1.6.0",
        "hf_xet.hf_xet",
        manifest="hf_xet/Cargo.toml",
        # tokio >= 1.53 uses stable std::os::fd on WASI; reqwest >= 0.13.5
        # distinguishes WASI from the browser.
        cargo_update=("tokio", "reqwest"),
        patches=(
            (
                "hf_xet/src/legacy/runtime.rs",
                '#[cfg(windows)]\nextern "system" fn console_ctrl_handler(',
                '#[cfg(target_os = "wasi")]\n'
                "fn install_sigint_handler() -> Result<(), RuntimeError> {\n"
                "    // A WebAssembly component receives no signals; SIGINT cannot arrive.\n"
                "    Ok(())\n"
                "}\n\n"
                '#[cfg(windows)]\nextern "system" fn console_ctrl_handler(',
            ),
        ),
        crate_patches=(
            OS_STR_BYTES_WASI,
            browser_wasm_only("reqwest-middleware", "0.5.1"),
            AWS_LC_SYS_WASI,
        ),
        # xet's wasm32 code paths target the browser (tokio_with_wasm, fetch).
        # WASI is not a browser: use the native tokio runtime paths.
        replace_all=tuple(
            (pattern, old, _browser)
            for pattern in ("**/*.rs", "**/Cargo.toml")
            for old, _browser in (
                ('target_family = "wasm"', 'all(target_family = "wasm", not(target_os = "wasi"))'),
                ('target_arch = "wasm32"', 'all(target_arch = "wasm32", not(target_os = "wasi"))'),
            )
        )
        + (WASI_PID,),
    ),
    "litellm": Rust(
        "litellm-1.102.1",
        "litellm.rust_bridge._native",
        manifest="litellm-rust/crates/python-bridge/Cargo.toml",
        workspace="litellm-rust/Cargo.toml",
        python_source=".",
        python_packages=("litellm",),
        features=("extension-module",),
        crate_patches=(
            RUSTLS_NATIVE_CERTS_WASI,
            GCP_AUTH_WASI,
            AZURE_IDENTITY_WASI,
            REQWEST_WASI,
        ),
        replace_all=(WASI_PID,),
        # Bedrock signing via the AWS SDK's current hyper-1.x HTTPS client
        # rather than the legacy hyper-0.14 client (socket2 0.5: no WASI).
        patches=(
            # The proxy's orphan reaper is Linux-only (/proc, prctl) and
            # returns early elsewhere, yet imports ctypes unconditionally.
            # CPython on WASI has no ctypes (no libffi): import it lazily.
            (
                "litellm/proxy/db/query_engine_reaper.py",
                "\nimport ctypes\nimport os\n",
                "\ntry:  # used only on Linux, after the platform check\n"
                "    import ctypes\n"
                "except ImportError:  # e.g. WASI: CPython without libffi\n"
                "    ctypes = None\n"
                "import os\n",
            ),
            # componentize-py bounds the bytes a single host call may copy
            # while snapshotting at build time; read the 2.4 MB bundled cost
            # map in chunks (identical result).
            (
                "litellm/litellm_core_utils/get_model_cost_map.py",
                '        return files("litellm").joinpath("model_prices_and_context_window_backup.json").read_bytes()\n',
                '        path = files("litellm").joinpath("model_prices_and_context_window_backup.json")\n'
                "        chunks = []\n"
                '        with path.open("rb") as handle:\n'
                "            while chunk := handle.read(1 << 18):\n"
                "                chunks.append(chunk)\n"
                '        return b"".join(chunks)\n',
            ),
            (
                "litellm-rust/crates/core/Cargo.toml",
                'aws-config = { version = "1.9.0", default-features = false, features = ["rustls", "rt-tokio"]',
                'aws-config = { version = "1.9.0", default-features = false, features = ["default-https-client", "rt-tokio"]',
            ),
            (
                "litellm-rust/crates/core/Cargo.toml",
                'aws-sdk-sts = { version = "1.108.0", default-features = false, features = ["rustls", "rt-tokio"]',
                'aws-sdk-sts = { version = "1.108.0", default-features = false, features = ["default-https-client", "rt-tokio"]',
            ),
        ),
    ),
    # --- C / Cython / mypyc ------------------------------------------------
    "regex": Pep517("regex-2026.9.10"),
    "markupsafe": Pep517("markupsafe-3.0.3"),
    "multidict": Pep517("multidict-6.9.1"),
    "frozenlist": Pep517("frozenlist-1.8.0"),
    "propcache": Pep517("propcache-0.5.4"),
    "yarl": Pep517("yarl-1.25.1"),
    "aiohttp": Pep517(
        "aiohttp-3.14.3",
        # llhttp's __wasm__ branch targets its JavaScript embedding (callbacks
        # imported from JS). WASI is a native C environment: use the C path.
        patches=(
            (
                "vendor/llhttp/src/native/api.c",
                "#if defined(__wasm__)\n",
                "#if defined(__wasm__) && !defined(__wasi__)\n",
            ),
        ),
    ),
    "charset-normalizer": Pep517(
        "charset_normalizer-3.5.1", env={"CHARSET_NORMALIZER_USE_CYTHON": "1"}
    ),
    "pyyaml": Pep517("pyyaml-6.0.3", env={"PYYAML_FORCE_LIBYAML": "1"}),
    "sqlalchemy": Pep517("sqlalchemy-2.1.1", env={"REQUIRE_SQLALCHEMY_CEXT": "1"}),
    # --- CPython stdlib ------------------------------------------------------
    "cpython-ssl": Stdlib(
        "cpython-wasi-ssl",
        "3.14.0",
        modules={"_ssl": "Modules/_ssl.c", "_hashlib": "Modules/_hashopenssl.c"},
        libs=("libssl.a", "libcrypto.a"),
    ),
    # --- meson -------------------------------------------------------------
    "numpy": Meson(
        "numpy-2.3.3",
        setup_args=(
            "-Dallow-noblas=true",  # numpy's bundled lapack-lite
            "-Dblas=none",
            "-Dlapack=none",
            "-Ddisable-threading=true",  # no threads in a component
            "-Ddisable-optimization=true",  # no x86/ARM SIMD dispatch on wasm32
            "-Ddisable-highway=true",
            "-Ddisable-svml=true",
            "-Ddisable-intel-sort=true",
            "-Dcpp_eh=none",  # wasi-sdk's libc++ is built without exceptions
            "-Db_lundef=false",  # libpython symbols bind at component link time
        ),
        patches=(
            # numpy only recognises wasm32 via emscripten's macro; the CPU is
            # the same under WASI.
            (
                "numpy/_core/include/numpy/npy_cpu.h",
                "#elif defined(__EMSCRIPTEN__)\n",
                "#elif defined(__EMSCRIPTEN__) || defined(__wasm__)\n",
            ),
            # pocketfft throws only on paths numpy's Python layer pre-validates
            # (n < 1, bad axes) or on allocation failure. Compile it like
            # numpy's own unique.cpp: -fexceptions, with throws resolved by
            # the cxa-terminate archive (see wasi/cxa_terminate.c).
            (
                "numpy/fft/meson.build",
                "  c_args: largefile_define,\n",
                "  c_args: largefile_define,\n  cpp_args: ['-fexceptions'],\n",
            ),
        ),
    ),
}


# ------------------------------------------------------------------ helpers


def run(cmd: list[str] | str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = cmd if isinstance(cmd, str) else shlex.join(cmd)
    print(f"+ {printable}", flush=True)
    subprocess.run(
        cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=True, shell=isinstance(cmd, str)
    )


def unpack(sdist: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    source = WORK / sdist
    if not source.exists():
        with tarfile.open(SDISTS / f"{sdist}.tar.gz") as archive:
            archive.extractall(WORK, filter="data")
    return source


def metadata(source: Path) -> tuple[str, str, bytes]:
    raw = (source / "PKG-INFO").read_bytes()
    parsed = email.parser.BytesParser().parsebytes(raw)
    return parsed["Name"], parsed["Version"], raw


def write_wheel(name: str, version: str, pkg_info: bytes, tree: Path) -> Path:
    """Zip ``tree`` into a spec-compliant wheel tagged for WASI CPython 3.14."""
    WHEELS.mkdir(parents=True, exist_ok=True)
    dist = re.sub(r"[-_.]+", "_", name)
    info = f"{dist}-{version}.dist-info"
    wheel_path = WHEELS / f"{dist}-{version}-{WHEEL_TAG}.whl"
    records = []
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as archive:

        def add(arcname: str, data: bytes) -> None:
            archive.writestr(arcname, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            records.append((arcname, f"sha256={digest.decode()}", str(len(data))))

        for path in sorted(tree.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path.relative_to(tree).as_posix(), path.read_bytes())
        add(f"{info}/METADATA", pkg_info)
        add(
            f"{info}/WHEEL",
            f"Wheel-Version: 1.0\nGenerator: dspy-wasm\nRoot-Is-Purelib: false\nTag: {WHEEL_TAG}\n".encode(),
        )
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(records)
        writer.writerow((f"{info}/RECORD", "", ""))
        archive.writestr(f"{info}/RECORD", buffer.getvalue())
    return wheel_path


# ------------------------------------------------------------------ rust


def pyo3_config() -> Path:
    path = BUILD / "wasi" / "pyo3-config.txt"
    path.write_text(
        "implementation=CPython\nversion=3.14\nshared=true\nabi3=false\n"
        f"lib_name=python3.14\nlib_dir={PYBUILD}\npointer_width=32\nbuild_flags=\n"
        "suppress_build_script_link_lines=false\n"
    )
    return path


def wire_crate_patches(source: Path, recipe: Rust) -> None:
    if not recipe.crate_patches:
        return
    registry = next((Path.home() / ".cargo" / "registry" / "src").glob("index.crates.io-*"))
    lines = []
    for patch in recipe.crate_patches:
        origin = registry / f"{patch.name}-{patch.version}"
        if not origin.exists():  # populate the registry cache
            run(["cargo", "fetch", "--manifest-path", str(source / recipe.manifest)])
        copy = BUILD / "wasi_crates" / f"{patch.name}-{patch.version}"
        if not copy.exists():
            shutil.copytree(origin, copy)
            apply_patches(copy, patch.edits)
            replace_everywhere(copy, patch.replace_all)
            for relative, content in patch.files.items():
                (copy / relative).write_text(content)
        lines.append(f'{patch.name} = {{ path = "{copy}" }}')
    workspace = source / (recipe.workspace or recipe.manifest)
    text = workspace.read_text()
    if "[patch.crates-io]" not in text:
        workspace.write_text(text + "\n[patch.crates-io]\n" + "\n".join(lines) + "\n")


def replace_everywhere(root: Path, rules: tuple[tuple[str, str, str], ...]) -> None:
    marker = root / ".wasi-replaced"
    if not rules or marker.exists():
        return
    for pattern, old, new in rules:
        for path in root.glob(pattern):
            text = path.read_text()
            if old in text:
                path.write_text(text.replace(old, new))
    marker.write_text("")


def build_rust(recipe: Rust) -> Path:
    source = unpack(recipe.sdist)
    apply_patches(source, recipe.patches)
    replace_everywhere(source, recipe.replace_all)
    wire_crate_patches(source, recipe)
    for crate in recipe.cargo_update:
        run(["cargo", "update", "-p", crate, "--manifest-path", str(source / recipe.manifest)])
    name, version, pkg_info = metadata(source)
    manifest = source / recipe.manifest
    crate = tomllib.loads(manifest.read_text())
    lib = crate.get("lib", {}).get("name") or crate["package"]["name"].replace("-", "_")

    cmd = [
        "cargo",
        "build",
        "--release",
        "--lib",
        "--target",
        "wasm32-wasip2",
        "--manifest-path",
        str(manifest),
    ]
    if recipe.features:
        cmd += ["--features", ",".join(recipe.features)]
    if recipe.no_default_features:
        cmd.append("--no-default-features")
    target_dir = BUILD / "wasi_cargo" / recipe.sdist
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RUSTFLAGS", "CARGO_BUILD"))}
    env.update(
        {
            # The componentize-py runtime recipe: a PIC shared library linked by
            # wasi-sdk clang against libpython3.14, never wrapped as a component.
            "RUSTFLAGS": (
                "--cfg pyo3_disable_reference_pool "
                # tokio's documented WASI mode (fs/net/... behind tokio_unstable)
                "--cfg tokio_unstable "
                "-Clink-args=-Wl,--skip-wit-component -Clink-args=-shared "
                f"-Clink-args=-L{PYBUILD} -Clink-args=-lpython3.14 -Clink-self-contained=n "
                + " ".join(f"-Clink-args={lib}" for lib in WASI_EMULATION_LIBS.split())
                + f" -Clink-args={cxa_terminate_archive()} {recipe.rustflags}"
            ),
            "CARGO_TARGET_WASM32_WASIP2_LINKER": f"{SDK}/bin/clang",
            "CARGO_TARGET_DIR": str(target_dir),
            "PYO3_CONFIG_FILE": str(pyo3_config()),
            "PYO3_CROSS_PYTHON_VERSION": "3.14",
            # Build scripts compiling C (onig, zstd, ring, ...) use wasi-sdk.
            "CC_wasm32_wasip2": f"{SDK}/bin/clang",
            "CXX_wasm32_wasip2": f"{SDK}/bin/clang++",
            "AR_wasm32_wasip2": f"{SDK}/bin/llvm-ar",
            "CFLAGS_wasm32_wasip2": f"--target=wasm32-wasip2 -fPIC {WASI_EMULATION_DEFINES}",
            # C++ inside crates (esaxx, ...) may use try/catch; throws resolve
            # to the cxa-terminate archive linked below.
            "CXXFLAGS_wasm32_wasip2": (
                f"--target=wasm32-wasip2 -fPIC -fexceptions {WASI_EMULATION_DEFINES}"
            ),
            "WASI_SDK_PATH": str(SDK),
            **recipe.env,
        }
    )
    run(cmd, cwd=source, env=env)
    artifact = target_dir / "wasm32-wasip2" / "release" / f"{lib}.wasm"

    tree = WORK / f"{recipe.sdist}-tree"
    shutil.rmtree(tree, ignore_errors=True)
    tree.mkdir(parents=True)
    *package_parts, module = recipe.module.split(".")
    if recipe.python_source is not None:
        for package in recipe.python_packages:
            shutil.copytree(
                source / recipe.python_source / package,
                tree / package,
                ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyd"),
            )
    else:
        # maturin's pure-Rust layout: a package re-exporting the extension.
        package_dir = tree.joinpath(*package_parts)
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "__init__.py").write_text(
            f"from .{module} import *\n\n__doc__ = {module}.__doc__\n"
            f'if hasattr({module}, "__all__"):\n    __all__ = {module}.__all__\n'
        )
        for stub in source.glob(f"**/{package_parts[-1]}.pyi"):
            shutil.copy(stub, package_dir / "__init__.pyi")
            break
        (package_dir / "py.typed").write_text("")
    destination = tree.joinpath(*package_parts, module + EXT_SUFFIX)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(artifact, destination)
    return write_wheel(name, version, pkg_info, tree)


# ------------------------------------------------------------------ pep 517


def cxa_terminate_archive() -> Path:
    """PIC archive with weak __cxa_throw/__cxa_allocate_exception (terminate)."""
    archive = BUILD / "wasi" / "lib" / "libwasi-cxa-terminate.a"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        obj = archive.with_suffix(".o")
        run(
            [
                f"{SDK}/bin/clang",
                "--target=wasm32-wasip2",
                "-fPIC",
                "-O2",
                "-c",
                str(ROOT / "wasi" / "cxa_terminate.c"),
                "-o",
                str(obj),
            ]
        )
        run([f"{SDK}/bin/llvm-ar", "rcs", str(archive), str(obj)])
    return archive


def cross_env() -> dict[str, str]:
    """Point setuptools/distutils at the WASI build instead of the host."""
    include = TC["WASI_PYTHON_INCLUDE"]
    cflags = f"-fPIC -O2 -I{include} -I{DEPS}/include {WASI_EMULATION_DEFINES} {PY_SHARED_ABI}"
    return {
        "_PYTHON_SYSCONFIGDATA_NAME": SYSCONFIG["__module__"],
        "_PYTHON_HOST_PLATFORM": "wasi-wasm32",
        "PYTHONPATH": str(SYSCONFIG_DIR),
        "CC": CLANG,
        "CXX": f"{SDK}/bin/clang++ --target=wasm32-wasip2 -fno-exceptions",
        "LDSHARED": f"{CLANG} -shared -L{PYBUILD} -L{DEPS}/lib -lpython3.14 {WASI_EMULATION_LIBS}",
        "LDCXXSHARED": (
            f"{SDK}/bin/clang++ --target=wasm32-wasip2 -shared -L{PYBUILD} -lpython3.14 "
            f"{WASI_EMULATION_LIBS} {cxa_terminate_archive()}"
        ),
        "AR": f"{SDK}/bin/llvm-ar",
        "CFLAGS": cflags,
        "CXXFLAGS": cflags,
        "LDFLAGS": "",
        "WASI_SDK_PATH": str(SDK),
    }


def apply_patches(source: Path, patches: tuple[tuple[str, str, str], ...]) -> None:
    for relative, old, new in patches:
        path = source / relative
        text = path.read_text()
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(f"patch target not found in {path}: {old!r}")
        path.write_text(text.replace(old, new, 1))


def grouping_tolerant_linker() -> Path:
    """wasm-ld behind a shim for meson.

    - drops --start-group/--end-group: meson treats LLD as GNU-style and
      brackets libraries; wasm-ld resolves archives order-independently.
    - for -shared links only, imports unresolved symbols dynamically (the
      libpython API, bound by componentize-py). Executables keep strict
      resolution so meson's has_function probes stay truthful.
    """
    path = BUILD / "wasi" / "bin" / "wasm-ld"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        "shared=\n"
        'for arg do shift; case "$arg" in --start-group|--end-group) ;; '
        '-shared) shared=1; set -- "$@" "$arg" ;; '
        '*) set -- "$@" "$arg" ;; esac; done\n'
        '[ -n "$shared" ] && set -- "$@" --unresolved-symbols=import-dynamic\n'
        f'exec "{SDK}/bin/wasm-ld" "$@"\n'
    )
    path.chmod(0o755)
    return path


def meson_cross_file() -> Path:
    """Describe the WASI target to meson (meson-python cross builds)."""
    path = BUILD / "wasi" / "wasm32-wasip2.meson"
    flags = ["--target=wasm32-wasip2", "-fPIC", *WASI_EMULATION_DEFINES.split(), PY_SHARED_ABI]
    linker = grouping_tolerant_linker()
    # libpython symbols stay undefined in the PIC module and are bound by
    # componentize-py's linker against the component's libpython3.14.so.
    link = [*WASI_EMULATION_LIBS.split(), str(cxa_terminate_archive())]

    def array(values: list[str]) -> str:
        return "[" + ", ".join(repr(v) for v in values) + "]"

    path.write_text(
        "[binaries]\n"
        f"c = {array([f'{SDK}/bin/clang', *flags])}\n"
        f"cpp = {array([f'{SDK}/bin/clang++', *flags, '-fno-exceptions'])}\n"
        f"c_ld = '{linker}'\ncpp_ld = '{linker}'\n"
        f"ar = '{SDK}/bin/llvm-ar'\n"
        f"strip = '{SDK}/bin/llvm-strip'\n"
        "[host_machine]\n"
        "system = 'wasi'\ncpu_family = 'wasm32'\ncpu = 'wasm32'\nendian = 'little'\n"
        "[properties]\n"
        "needs_exe_wrapper = true\n"
        "longdouble_format = 'IEEE_QUAD_LE'\n"  # wasm32 long double is binary128
        "[built-in options]\n"
        f"c_link_args = {array(link)}\n"
        f"cpp_link_args = {array(link)}\n"
    )
    return path


def build_venv(source: Path, recipe: Pep517) -> str:
    """Host-side build environment, created *without* the cross variables.

    pip's own isolation would resolve build tools (Cython, ninja, patchelf)
    for the target platform, i.e. try to compile them for WebAssembly.
    """
    venv = BUILD / "wasi_venvs" / recipe.sdist
    python = venv / "bin" / "python"
    if not python.exists():
        requires = tomllib.loads((source / "pyproject.toml").read_text())["build-system"][
            "requires"
        ]
        if isinstance(recipe, Meson):
            requires = [*requires, "ninja"]
        run(["uv", "venv", "--seed", "--python", HOST_PYTHON, str(venv)])
        run(["uv", "pip", "install", "--python", str(python), "wheel", *requires])
    return str(python)


def build_pep517(recipe: Pep517) -> Path:
    source = unpack(recipe.sdist)
    apply_patches(source, recipe.patches)
    out = WORK / f"{recipe.sdist}-dist"
    shutil.rmtree(out, ignore_errors=True)
    python = build_venv(source, recipe)
    cmd = [
        python,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(out),
        str(source),
    ]
    settings = list(recipe.config_settings)
    if isinstance(recipe, Meson):
        settings.append(f"setup-args=--cross-file={meson_cross_file()}")
        settings += [f"setup-args={arg}" for arg in recipe.setup_args]
    for setting in settings:
        cmd += ["--config-settings", setting]
    tools = str(Path(python).parent)  # cython, ninja, meson from the build venv
    run(cmd, env={**cross_env(), "PATH": f"{tools}:{os.environ['PATH']}", **recipe.env})
    built = next(out.glob("*.whl"))
    # Re-tag: the backend saw a wasi platform; normalise to the canonical tag.
    name, version, _ = metadata(source)
    tree = WORK / f"{recipe.sdist}-tree"
    shutil.rmtree(tree, ignore_errors=True)
    with zipfile.ZipFile(built) as archive:
        archive.extractall(tree)
    info = next(tree.glob("*.dist-info"))
    pkg_info = (info / "METADATA").read_bytes()
    shutil.rmtree(info)
    extensions = [p for p in tree.rglob("*.so")]
    if not extensions:
        raise RuntimeError(f"{recipe.sdist}: backend produced no extension modules")
    for ext in extensions:
        magic = ext.read_bytes()[:4]
        if magic != b"\0asm":
            raise RuntimeError(f"{ext} is not WebAssembly")
    return write_wheel(name, version, pkg_info, tree)


# ------------------------------------------------------------------ stdlib


def build_stdlib(recipe: Stdlib) -> Path:
    src = Path(TC["CPYTHON_SRC"])
    tree = WORK / f"{recipe.name}-tree"
    shutil.rmtree(tree, ignore_errors=True)
    tree.mkdir(parents=True)
    flags = [
        "--target=wasm32-wasip2",
        "-fPIC",
        "-O2",
        "-DPy_BUILD_CORE_MODULE",
        PY_SHARED_ABI,
        *WASI_EMULATION_DEFINES.split(),
        f"-I{src}/Include",
        f"-I{src}/Include/internal",
        f"-I{PYBUILD}",
        f"-I{DEPS}/include",
    ]
    for module, source in recipe.modules.items():
        obj = tree / f"{module}.o"
        run([f"{SDK}/bin/clang", *flags, "-c", str(src / source), "-o", str(obj)])
        # -lc after the static archives: they reference libc symbols (sockets).
        run(
            [
                f"{SDK}/bin/clang",
                "--target=wasm32-wasip2",
                "-shared",
                str(obj),
                f"-L{PYBUILD}",
                "-lpython3.14",
                *(str(DEPS / "lib" / lib) for lib in recipe.libs),
                "-lc",
                *WASI_EMULATION_LIBS.split(),
                "-o",
                str(tree / f"{module}{EXT_SUFFIX}"),
            ]
        )
        obj.unlink()
    pkg_info = (
        f"Metadata-Version: 2.1\nName: {recipe.name}\nVersion: {recipe.version}\n"
        f"Summary: CPython {recipe.version} {', '.join(recipe.modules)} for wasm32-wasip2\n"
    ).encode()
    return write_wheel(recipe.name, recipe.version, pkg_info, tree)


# ------------------------------------------------------------------ commands


def build(names: list[str], missing_only: bool = False) -> None:
    for name in names or list(RECIPES):
        marker = WHEELS / f".{name}.built"
        if missing_only and marker.exists() and (WHEELS / marker.read_text().strip()).exists():
            print(f"CACHED {name}: {marker.read_text().strip()}", flush=True)
            continue
        recipe = RECIPES[name]
        if isinstance(recipe, Rust):
            wheel = build_rust(recipe)
        elif isinstance(recipe, Stdlib):
            wheel = build_stdlib(recipe)
        else:
            wheel = build_pep517(recipe)
        marker.write_text(wheel.name)
        print(f"BUILT {name}: {wheel.name}", flush=True)


def fetch() -> None:
    """Download the pinned sdists (wasi/native.txt) from PyPI."""
    import json
    import urllib.request

    SDISTS.mkdir(parents=True, exist_ok=True)
    for spec in (ROOT / "wasi" / "native.txt").read_text().split():
        name, version = spec.split("==")
        meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json"))
        sdist = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
        target = SDISTS / sdist["filename"]
        if not target.exists():
            urllib.request.urlretrieve(sdist["url"], target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != sdist["digests"]["sha256"]:
                target.unlink()
                sys.exit(f"sha256 mismatch for {sdist['filename']}")
        print(f"FETCHED {sdist['filename']}")


def install() -> None:
    """Assemble build/wasi_deps: every locked package, native ones from our wheels."""
    lock = [
        line.split("#")[0].strip()
        for line in (ROOT / "wasi" / "requirements.lock").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    native = {
        re.sub(r"[-_.]+", "-", wheel.name.split("-")[0]).lower(): wheel
        for wheel in sorted(WHEELS.glob("*.whl"))
    }
    pure = [
        spec for spec in lock if re.sub(r"[-_.]+", "-", spec.split("==")[0]).lower() not in native
    ]
    missing = [
        spec
        for spec in lock
        if re.sub(r"[-_.]+", "-", spec.split("==")[0]).lower() not in native
        and spec.split("==")[0].lower().replace("_", "-") in {r.lower() for r in RECIPES}
    ]
    if missing:
        sys.exit(f"native wheels not built: {missing}")
    shutil.rmtree(TARGET, ignore_errors=True)
    TARGET.mkdir(parents=True)
    run(
        [
            HOST_PYTHON,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--target",
            str(TARGET),
            "--only-binary",
            ":all:",
            "--platform",
            "any",
            "--python-version",
            "3.14",
            "--implementation",
            "py",
            *pure,
        ]
    )
    for wheel in native.values():
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(TARGET)
    print(f"INSTALLED {len(pure)} pure + {len(native)} native packages into {TARGET}")


def main() -> None:
    command, *names = sys.argv[1:] or ["build"]
    if command == "build":
        missing = "--missing" in names
        build([n for n in names if n != "--missing"], missing_only=missing)
    elif command == "install":
        install()
    elif command == "fetch":
        fetch()
    elif command == "suffix":
        print(EXT_SUFFIX)
    else:
        sys.exit(f"unknown command {command}")


if __name__ == "__main__":
    main()
