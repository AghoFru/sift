#include "sift.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static void require_ok(int32_t status) {
    if (status != SIFT_OK) {
        fprintf(stderr, "Sift failed: %s\n", sift_last_error());
        exit(1);
    }
}

static void expect_search(Sift *handle, const char *query, const char *fragment) {
    char *result = NULL;
    require_ok(sift_search(handle, sift_text(query), &result));
    assert(result != NULL);
    if (strstr(result, fragment) == NULL) {
        fprintf(stderr, "Unexpected search result: %s\n", result);
        sift_string_free(result);
        exit(1);
    }
    sift_string_free(result);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: c-api-check LOCAL_MODEL NEW_INDEX\n");
        return 2;
    }
    assert(sift_abi_version() == 1);
    const char *documents = "["
        "{\"id\":\"cat\",\"text\":\"A domestic cat sleeps on the windowsill.\"},"
        "{\"id\":\"dog\",\"text\":\"A dog plays in the park.\"},"
        "{\"id\":\"bird\",\"text\":\"A sparrow flies over a tree.\"}]";
    Sift *handle = NULL;
    require_ok(sift_create(sift_text(argv[2]), sift_text(argv[1]), sift_text(documents), &handle));
    const char *cat = "{\"q\":\"cat\",\"blend_alpha\":0}";
    const char *horse = "{\"q\":\"horse\",\"blend_alpha\":0}";
    expect_search(handle, cat, "\"doc_id\":\"cat\"");
    require_ok(sift_write(handle, sift_text(
        "[{\"id\":\"cat\",\"text\":\"A horse rests near a stable.\"}]"), SIFT_UPSERT));
    expect_search(handle, cat, "\"hits\":[]");
    expect_search(handle, horse, "\"doc_id\":\"cat\"");
    require_ok(sift_delete(handle, sift_text("[\"dog\"]")));
    require_ok(sift_compact(handle));
    require_ok(sift_reload(handle));
    assert(sift_write(handle, sift_text("[]"), 99) == SIFT_ERROR);
    assert(strlen(sift_last_error()) > 0);
    char *invalid_result = (char *)&handle;
    assert(sift_search(handle, sift_text("{bad json"), &invalid_result) == SIFT_ERROR);
    assert(invalid_result == NULL);
    expect_search(handle, horse, "\"doc_id\":\"cat\"");
    sift_close(handle);
    handle = NULL;
    require_ok(sift_open(sift_text(argv[2]), &handle));
    expect_search(handle, horse, "\"doc_id\":\"cat\"");
    expect_search(handle, "{\"q\":\"dog\",\"blend_alpha\":0}", "\"hits\":[]");
    sift_close(handle);
    sift_close(NULL);
    sift_string_free(NULL);
    assert(sift_search(NULL, sift_text(cat), &invalid_result) == SIFT_ERROR);
    assert(invalid_result == NULL);
    puts("C create, search, upsert, delete, compact, reload, reopen, and error checks passed.");
    return 0;
}
