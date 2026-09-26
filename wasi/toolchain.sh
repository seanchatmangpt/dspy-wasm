#!/usr/bin/env bash
# Reproducible WASI toolchain for building CPython 3.14 native extensions that
# load into componentize-py 0.25.1 components (wasm32-wasip2, PIC, shared).
#
# Mirrors componentize-py's own build.rs: wasi-sdk 33, CPython 3.14.0
# configured for --host=wasm32-unknown-wasip2 with -fPIC, zlib from source.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${WASI_BUILD_DIR:-$ROOT/build/wasi}"
WASI_SDK_VERSION="${WASI_SDK_VERSION:-33}"
PYTHON_VERSION="${PYTHON_VERSION:-3.14.0}"
ZLIB_VERSION="${ZLIB_VERSION:-1.3.1}"
JOBS="${JOBS:-$(nproc)}"

mkdir -p "$OUT"
cd "$OUT"

# fetch URL DEST SHA256: download with retries and verify the pinned digest,
# so an error page from a mirror fails here rather than as a tar error later.
fetch() {
  curl -sSfL --retry 5 --retry-all-errors -o "$2" "$1"
  if ! echo "$3  $2" | sha256sum -c --quiet -; then
    echo "sha256 mismatch for $1" >&2
    rm -f "$2"
    exit 1
  fi
}

# --- wasi-sdk -----------------------------------------------------------------
SDK="$OUT/wasi-sdk"
if [ ! -x "$SDK/bin/clang" ]; then
  fetch "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_SDK_VERSION}/wasi-sdk-${WASI_SDK_VERSION}.0-x86_64-linux.tar.gz" \
    wasi-sdk.tar.gz 0ba8b5bfaeb2adf3f29bab5841d76cf5318ab8e1642ea195f88baba1abd47bce
  rm -rf "$SDK" && mkdir -p "$SDK"
  tar -xzf wasi-sdk.tar.gz -C "$SDK" --strip-components=1
  rm wasi-sdk.tar.gz
fi

# --- host python (same major.minor as the component) --------------------------
# The interpreter lives under $OUT too: a venv only symlinks to it, so a cached
# build/wasi restored on another machine would otherwise hold a dangling link.
HOST_PY="$OUT/host-python"
export UV_PYTHON_INSTALL_DIR="$OUT/uv-python"
if [ ! -x "$HOST_PY/bin/python" ]; then
  uv venv --clear --seed --managed-python --python "$PYTHON_VERSION" "$HOST_PY"
fi

# --- CPython source -----------------------------------------------------------
SRC="$OUT/cpython"
if [ ! -f "$SRC/configure" ]; then
  fetch "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
    cpython.tar.xz 2299dae542d395ce3883aca00d3c910307cd68e0b2f7336098c8e7b7eee9f3e9
  rm -rf "$SRC" && mkdir -p "$SRC"
  tar -xJf cpython.tar.xz -C "$SRC" --strip-components=1
  rm cpython.tar.xz
fi

# Native build python (configure requires an exact-version build interpreter).
NATIVE="$SRC/builddir/build"
if [ ! -x "$NATIVE/python" ]; then
  mkdir -p "$NATIVE"
  (cd "$NATIVE" && ../../configure --prefix="$NATIVE/install" >/dev/null && make -j"$JOBS" >/dev/null)
fi

WASI="$SRC/builddir/wasi"
DEPS="$WASI/deps"
mkdir -p "$WASI" "$DEPS"

# --- zlib for wasm32-wasip2 (PIC) ---------------------------------------------
if [ ! -f "$DEPS/lib/libz.a" ]; then
  fetch "https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz" \
    zlib.tar.gz 9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
  rm -rf zlib && mkdir zlib && tar -xzf zlib.tar.gz -C zlib --strip-components=1 && rm zlib.tar.gz
  (cd zlib && CC="$SDK/bin/clang --target=wasm32-wasip2" CFLAGS="-fPIC -O2" AR="$SDK/bin/llvm-ar" \
     RANLIB="$SDK/bin/llvm-ranlib" ./configure --static --prefix="$DEPS" >/dev/null && make -j"$JOBS" install >/dev/null)
fi

# --- libyaml for wasm32-wasip2 (PIC), linked statically into PyYAML's _yaml ----
LIBYAML_VERSION="${LIBYAML_VERSION:-0.2.5}"
if [ ! -f "$DEPS/lib/libyaml.a" ]; then
  fetch "https://github.com/yaml/libyaml/releases/download/${LIBYAML_VERSION}/yaml-${LIBYAML_VERSION}.tar.gz" \
    yaml.tar.gz c642ae9b75fee120b2d96c712538bd2cf283228d2337df2cf2988e3c02678ef4
  rm -rf libyaml && mkdir libyaml && tar -xzf yaml.tar.gz -C libyaml --strip-components=1 && rm yaml.tar.gz
  # libyaml's bundled config.sub predates WASI; its sources need no configure.
  (
    cd libyaml && mkdir -p obj
    for c in src/*.c; do
      "$SDK/bin/clang" --target=wasm32-wasip2 -fPIC -O2 -Iinclude -DYAML_DECLARE_STATIC \
        -DYAML_VERSION_MAJOR=0 -DYAML_VERSION_MINOR=2 -DYAML_VERSION_PATCH=5 \
        -DYAML_VERSION_STRING="\"${LIBYAML_VERSION}\"" -c "$c" -o "obj/$(basename "$c" .c).o"
    done
    mkdir -p "$DEPS/lib" "$DEPS/include"
    "$SDK/bin/llvm-ar" rcs "$DEPS/lib/libyaml.a" obj/*.o
    cp include/yaml.h "$DEPS/include/"
  )
fi

# --- OpenSSL for wasm32-wasip2 (static, PIC), backing CPython's _ssl/_hashlib --
# Thread-safe (OPENSSL_THREADS, as CPython requires) over wasi-libc's
# single-threaded pthreads; no thread pool (it would create threads), no
# Unix-domain sockets and no socketpair-based QUIC notifier (absent on WASI).
OPENSSL_VERSION="${OPENSSL_VERSION:-3.5.4}"
if [ ! -f "$DEPS/lib/libssl.a" ]; then
  fetch "https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz" \
    openssl.tar.gz 967311f84955316969bdb1d8d4b983718ef42338639c621ec4c34fddef355e99
  rm -rf openssl && mkdir openssl && tar -xzf openssl.tar.gz -C openssl --strip-components=1 && rm openssl.tar.gz
  python3 - openssl/crypto/thread/arch/thread_posix.c <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = "int ossl_crypto_thread_native_exit(void)\n{\n    pthread_exit(NULL);\n"
new = ("int ossl_crypto_thread_native_exit(void)\n{\n"
       "#ifndef __wasi__  /* WASI: no threads besides the caller can exist */\n"
       "    pthread_exit(NULL);\n#endif\n")
assert old in text or new in text
open(path, "w").write(text.replace(old, new))
PY
  (
    cd openssl
    CC="$SDK/bin/clang" AR="$SDK/bin/llvm-ar" RANLIB="$SDK/bin/llvm-ranlib" \
    CFLAGS="--target=wasm32-wasip2 -fPIC -O2 -DOPENSSL_NO_UNIX_SOCK -DRIO_NOTIFIER_METHOD=1 \
      -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_GETPID -D_WASI_EMULATED_MMAN" \
    ./Configure linux-generic32 --prefix="$DEPS" --libdir=lib --openssldir=/etc/ssl \
      no-shared no-module no-dso no-asm no-async no-secure-memory no-afalgeng \
      no-ui-console no-tests no-apps no-docs no-thread-pool no-default-thread-pool >/dev/null
    make -j"$JOBS" build_libs >/dev/null
    make install_dev >/dev/null
  )
fi

# --- CPython for wasm32-wasip2 ------------------------------------------------
if [ ! -f "$WASI/libpython3.14.a" ]; then
  (
    cd "$WASI"
    CONFIG_SITE=../../Tools/wasm/wasi/config.site-wasm32-wasi \
    WASI_SDK_PATH="$SDK" \
    CFLAGS="--target=wasm32-wasip2 -fPIC -I$DEPS/include" \
    LDFLAGS="--target=wasm32-wasip2 -L$DEPS/lib" \
    ../../Tools/wasm/wasi-env ../../configure -C \
      --host=wasm32-unknown-wasip2 \
      --build="$(../../config.guess)" \
      --with-build-python="$NATIVE/python" \
      --prefix="$WASI/install" \
      --disable-test-modules \
      --enable-ipv6 >/dev/null
    make -j"$JOBS" build_all >/dev/null
    make install >/dev/null
  )
fi

# Shared libpython for link-time symbol resolution by extension builds.
if [ ! -f "$WASI/libpython3.14.so" ]; then
  "$SDK/bin/clang" --target=wasm32-wasip2 -shared -o "$WASI/libpython3.14.so" \
    -Wl,--whole-archive "$WASI/libpython3.14.a" -Wl,--no-whole-archive \
    "$WASI"/Modules/_hacl/*.a "$WASI/Modules/_decimal/libmpdec/libmpdec.a" \
    "$WASI/Modules/expat/libexpat.a" "$DEPS/lib/libz.a" \
    -lwasi-emulated-signal -lwasi-emulated-getpid -lwasi-emulated-process-clocks -ldl
fi

# --- componentize-py 0.25.1 with a build-time hostcall budget for large apps ---
# Built from the published crate (which ships the same prebuilt runtime, libc
# and libpython artifacts as the PyPI wheel). One change: pre-initialisation
# runs the application being packaged, so the guest is trusted and its whole
# linear memory must be retrievable in one host call. wasmtime's default
# 128 MiB DoS guard is too small once litellm's import graph is resident.
COMPONENTIZE_PY_VERSION="${COMPONENTIZE_PY_VERSION:-0.25.1}"
CPY="$OUT/bin/componentize-py"
if [ ! -x "$CPY" ]; then
  rm -rf componentize-py-src && mkdir componentize-py-src
  fetch "https://static.crates.io/crates/componentize-py/componentize-py-${COMPONENTIZE_PY_VERSION}.crate" \
    componentize-py.crate addb8010b8a05a736a7efaee36975e616bdc4762c458a3ad553405354b90ed47
  tar -xzf componentize-py.crate -C componentize-py-src --strip-components=1 && rm componentize-py.crate
  python3 - componentize-py-src/src/lib.rs <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = "        let mut store = Store::new(&engine, Ctx { wasi, table });\n"
new = old + (
    "        // Pre-initialisation runs the (trusted) application being packaged;\n"
    "        // its whole linear memory must fit in one host call.\n"
    "        store.set_hostcall_fuel(4 << 30);\n"
)
assert text.count(old) == 1
open(path, "w").write(text.replace(old, new))
PY
  (cd componentize-py-src && cargo build --release --bin componentize-py >/dev/null 2>&1)
  mkdir -p "$OUT/bin"
  cp componentize-py-src/target/release/componentize-py "$CPY"
  # The crate ships test SDKs with componentize-py.toml files; left under the
  # repo, `componentize-py -p .` would discover their WIT worlds.
  rm -rf componentize-py-src
fi

SYSCONFIG_DIR="$(cat "$WASI/pybuilddir.txt")"

# Make the WASI sysconfig importable by the host interpreter (as crossenv does),
# including inside pip's isolated build environments. It is inert unless
# _PYTHON_SYSCONFIGDATA_NAME selects it.
HOST_STDLIB="$("$HOST_PY/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])')"
cp "$WASI/$SYSCONFIG_DIR"/_sysconfigdata_*.py "$HOST_STDLIB/"
cat > "$OUT/env.sh" <<ENV
# Source to cross-compile extensions for componentize-py 0.25.1 components.
export WASI_SDK_PATH="$SDK"
export WASI_PYTHON_BUILD="$WASI"
export WASI_DEPS="$DEPS"
export WASI_PYTHON_INCLUDE="$WASI/install/include/python3.14"
export WASI_SYSCONFIG_DIR="$WASI/$SYSCONFIG_DIR"
export HOST_PYTHON="$HOST_PY/bin/python"
export CPYTHON_SRC="$SRC"
export COMPONENTIZE_PY="$CPY"
ENV
echo "toolchain ready: $OUT/env.sh"
