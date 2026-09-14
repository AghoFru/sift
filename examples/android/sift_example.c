#include <jni.h>
#include <limits.h>
#include "sift.h"

JNIEXPORT jbyteArray JNICALL Java_org_sift_example_OfflineCheck_search(
    JNIEnv *env, jclass klass, jbyteArray path, jbyteArray request) {
    (void)klass;
    if (!path || !request) {
        jclass exception = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
        if (exception) (*env)->ThrowNew(env, exception, "Path and request are required.");
        return NULL;
    }
    jbyte *path_bytes = (*env)->GetByteArrayElements(env, path, NULL);
    if (!path_bytes) return NULL;
    jbyte *request_bytes = (*env)->GetByteArrayElements(env, request, NULL);
    if (!request_bytes) {
        (*env)->ReleaseByteArrayElements(env, path, path_bytes, JNI_ABORT);
        return NULL;
    }
    SiftString index_path = {(const uint8_t *)path_bytes,
                            (size_t)(*env)->GetArrayLength(env, path)};
    SiftString query = {(const uint8_t *)request_bytes,
                       (size_t)(*env)->GetArrayLength(env, request)};
    Sift *handle = NULL;
    char *response = NULL;
    int32_t status = sift_open(index_path, &handle);
    if (status == SIFT_OK) status = sift_search(handle, query, &response);
    jbyteArray result = NULL;
    if (status == SIFT_OK && strlen(response) <= INT_MAX) {
        jsize length = (jsize)strlen(response);
        result = (*env)->NewByteArray(env, length);
        if (result) (*env)->SetByteArrayRegion(env, result, 0, length, (const jbyte *)response);
    } else {
        jclass exception = (*env)->FindClass(env, "java/lang/IllegalStateException");
        // Sift messages can contain UTF-8 paths. ThrowNew requires modified UTF-8.
        // Use a fixed ASCII message here. Native callers can inspect sift_last_error().
        if (exception) (*env)->ThrowNew(env, exception, "The native Sift operation failed.");
    }
    sift_string_free(response);
    sift_close(handle);
    (*env)->ReleaseByteArrayElements(env, request, request_bytes, JNI_ABORT);
    (*env)->ReleaseByteArrayElements(env, path, path_bytes, JNI_ABORT);
    return result;
}
