// WASI port of AWS-LC's console (password prompt) support.
//
// WASI has no controlling terminal, /dev/tty, termios or signal handlers.
// This follows upstream's own non-tty path (AWSLC_CONSOLE_NO_TTY_DETECT):
// prompts go to stderr, input is read from stdin, echo cannot be toggled.
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <openssl/err.h>
#include <openssl/pem.h>

#include "internal.h"
#include "../internal.h"

static struct CRYPTO_STATIC_MUTEX console_global_mutex = CRYPTO_STATIC_MUTEX_INIT;

void openssl_console_acquire_mutex(void) {
    CRYPTO_STATIC_MUTEX_lock_write(&console_global_mutex);
}

void openssl_console_release_mutex(void) {
    CRYPTO_STATIC_MUTEX_unlock_write(&console_global_mutex);
}

int openssl_console_open(void) { return 1; }

int openssl_console_close(void) { return 1; }

int openssl_console_write(const char *str) {
    if (fputs(str, stderr) < 0 || fflush(stderr) != 0) {
        OPENSSL_PUT_ERROR(PEM, PEM_R_PROBLEMS_GETTING_PASSWORD);
        if (ferror(stderr)) {
            ERR_add_error_data(2, "System error: ", strerror(errno));
            clearerr(stderr);
        }
        return 0;
    }
    return 1;
}

static int discard_line_remainder(FILE *in) {
    char buf[5];
    do {
        if (!fgets(buf, 4, in)) {
            if (ferror(in)) {
                OPENSSL_PUT_ERROR(PEM, PEM_R_PROBLEMS_GETTING_PASSWORD);
                ERR_add_error_data(2, "System error: ", strerror(errno));
                clearerr(in);
                return 0;
            }
            return feof(in) ? 1 : 0;
        }
    } while (strchr(buf, '\n') == NULL);
    return 1;
}

int openssl_console_read(char *buf, int minsize, int maxsize, int echo) {
    (void)echo;  // no terminal: echo cannot be disabled
    if (!buf || maxsize < minsize) {
        return -1;
    }
    buf[0] = '\0';
    char *p = fgets(buf, maxsize, stdin);
    if (p == NULL || feof(stdin) || ferror(stdin)) {
        OPENSSL_PUT_ERROR(PEM, PEM_R_PROBLEMS_GETTING_PASSWORD);
        if (ferror(stdin)) {
            ERR_add_error_data(2, "System error: ", strerror(errno));
            clearerr(stdin);
        }
        return -1;
    }
    if ((p = strchr(buf, '\n')) != NULL) {
        *p = '\0';
    } else if (!discard_line_remainder(stdin)) {
        return -1;
    }
    size_t input_len = strlen(buf);
    if (input_len < (size_t)minsize || input_len > (size_t)MAX_PASSWORD_LENGTH) {
        return -1;
    }
    return 0;
}
