/*
 * C++ throw support for a no-exceptions runtime.
 *
 * componentize-py links wasi-sdk's no-exceptions libc++/libc++abi into every
 * component, so a thrown C++ exception can never be caught there. Some
 * extensions nevertheless compile a few translation units with -fexceptions
 * (numpy's np.unique hashing, pocketfft); clang then lowers `throw` to these
 * two calls and drops the catch blocks. This archive gives them the contract
 * C++ specifies when exceptions are unavailable: report and terminate.
 * The symbols are weak, so a runtime with real exception support wins.
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((weak)) void *__cxa_allocate_exception(size_t size) {
    return malloc(size);
}

__attribute__((weak, noreturn)) void __cxa_throw(void *object, void *type_info, void (*destructor)(void *)) {
    (void)object; (void)type_info; (void)destructor;
    fputs("C++ exception thrown in a no-exceptions WebAssembly runtime; terminating\n", stderr);
    abort();
}
