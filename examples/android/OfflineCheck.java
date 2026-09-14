package org.sift.example;

import android.app.Activity;
import android.app.Instrumentation;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.json.JSONArray;
import org.json.JSONObject;

public final class OfflineCheck extends Instrumentation {
    static {
        System.loadLibrary("sift_example");
    }

    private static native byte[] search(byte[] path, byte[] request);

    @Override
    public void onCreate(Bundle arguments) {
        super.onCreate(arguments);
        start();
    }

    @Override
    public void onStart() {
        Bundle result = new Bundle();
        try {
            Context context = getTargetContext();
            if (context.checkSelfPermission("android.permission.INTERNET")
                    != PackageManager.PERMISSION_DENIED) {
                throw new AssertionError("Internet permission must be denied.");
            }
            File index = new File(context.getFilesDir(), "index");
            copyAsset(context, "index", index);
            byte[] path = index.getAbsolutePath().getBytes(StandardCharsets.UTF_8);
            byte[] request = "{\"q\":\"sparrow\",\"blend_alpha\":0}"
                    .getBytes(StandardCharsets.UTF_8);
            // Each call opens and closes the native index and frees its result.
            for (int call = 0; call < 3; call++) {
                JSONObject response = new JSONObject(new String(
                        search(path, request), StandardCharsets.UTF_8));
                JSONArray hits = response.getJSONArray("hits");
                if (hits.length() != 1
                        || !hits.getJSONObject(0).getString("doc_id").equals("bird")) {
                    throw new AssertionError("Unexpected offline search result: " + response);
                }
            }
            try {
                search(path, "{bad json".getBytes(StandardCharsets.UTF_8));
                throw new AssertionError("Invalid JSON must fail.");
            } catch (IllegalStateException expected) {
                if (expected.getMessage() == null || expected.getMessage().isEmpty()) {
                    throw new AssertionError("Native errors must include a message.");
                }
            }
            result.putString("stream", "SIFT_OFFLINE_OK: search, reopen, and errors passed.\n");
            finish(Activity.RESULT_OK, result);
        } catch (Exception | AssertionError failure) {
            result.putString("stream", "SIFT_OFFLINE_FAILED: " + failure + "\n");
            finish(Activity.RESULT_CANCELED, result);
        }
    }

    private static void copyAsset(Context context, String asset, File destination)
            throws IOException {
        String[] children = context.getAssets().list(asset);
        if (children == null) {
            throw new IOException("Cannot list asset: " + asset);
        }
        if (children.length > 0) {
            if (!destination.isDirectory() && !destination.mkdirs()) {
                throw new IOException("Cannot create directory: " + destination);
            }
            for (String child : children) {
                copyAsset(context, asset + "/" + child, new File(destination, child));
            }
            return;
        }
        try (InputStream input = context.getAssets().open(asset);
                FileOutputStream output = new FileOutputStream(destination)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
        }
    }
}
