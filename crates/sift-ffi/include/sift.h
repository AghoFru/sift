#ifndef SIFT_H
#define SIFT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Sift Sift;
typedef struct {
    const uint8_t *data;
    size_t len;
} SiftString;

enum { SIFT_OK = 0, SIFT_ERROR = 1, SIFT_PANIC = 2 };
enum { SIFT_INSERT = 0, SIFT_UPSERT = 1 };

/* Input bytes must remain valid and unchanged for the complete call. */
static inline SiftString sift_text(const char *value) {
    SiftString result = {(const uint8_t *)value, value ? strlen(value) : 0};
    return result;
}

uint32_t sift_abi_version(void);
/* Borrowed thread-local string, valid until the next fallible call on this thread. Do not free it. */
const char *sift_last_error(void);

/* On failure, output handle and string slots are set to NULL. */
int32_t sift_open(SiftString path, Sift **output);
int32_t sift_create(SiftString path, SiftString local_model, SiftString documents, Sift **output);
int32_t sift_search(Sift *handle, SiftString request, char **output);
int32_t sift_write(Sift *handle, SiftString documents, int32_t mode);
int32_t sift_delete(Sift *handle, SiftString identifiers);
int32_t sift_compact(Sift *handle);
int32_t sift_reload(Sift *handle);

/* Calls on one handle serialize. Do not close a handle while another call uses it. */
void sift_close(Sift *handle);
/* Free each successful search result once. Never use free() on a Sift result. */
void sift_string_free(char *value);

#ifdef __cplusplus
}
#endif
#endif
