//! C interface with explicit input lengths and owned result strings.

use anyhow::{anyhow, Context, Result};
use sift::{Engine, SearchOptions, WriteMode};
use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;

const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;
const MAX_PATH_BYTES: usize = 4096;

/// A borrowed UTF-8 input. Its bytes must remain valid for the complete call.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SiftString {
    pub data: *const u8,
    pub len: usize,
}

/// Opaque to C callers. Operations on one live handle serialize through its mutex.
pub struct Sift {
    engine: Mutex<Engine>,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn save_error(message: &str) {
    let message = CString::new(message.replace('\0', "\\0")).expect("Null bytes were replaced.");
    LAST_ERROR.with(|error| *error.borrow_mut() = message);
}

fn boundary(operation: impl FnOnce() -> Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => {
            save_error("");
            0
        }
        Ok(Err(error)) => {
            save_error(&format!("{error:#}"));
            1
        }
        Err(_) => {
            save_error("Sift caught an internal panic. Close and reopen the handle.");
            2
        }
    }
}

unsafe fn text<'a>(value: SiftString, limit: usize) -> Result<&'a str> {
    if value.len > limit {
        return Err(anyhow!("Input exceeds the {limit}-byte limit."));
    }
    if value.len == 0 {
        return Ok("");
    }
    if value.data.is_null() {
        return Err(anyhow!("A nonempty input requires a valid data pointer."));
    }
    // The C contract requires valid immutable bytes for the supplied length.
    let bytes = unsafe { std::slice::from_raw_parts(value.data, value.len) };
    std::str::from_utf8(bytes).context("Input must contain valid UTF-8.")
}

unsafe fn with_engine(
    handle: *mut Sift,
    operation: impl FnOnce(&mut Engine) -> Result<()>,
) -> Result<()> {
    // A non-null handle must have come from this library and must still be open.
    let handle = unsafe { handle.as_ref() }.context("The Sift handle is null.")?;
    let mut engine = handle
        .engine
        .lock()
        .map_err(|_| anyhow!("The engine lock is poisoned. Close and reopen the handle."))?;
    operation(&mut engine)
}

/// Return the C interface version.
#[no_mangle]
pub extern "C" fn sift_abi_version() -> u32 {
    1
}

/// Return the current thread's error string. Do not free this borrowed pointer.
#[no_mangle]
pub extern "C" fn sift_last_error() -> *const c_char {
    LAST_ERROR.with(|error| error.borrow().as_ptr())
}

/// Open an existing index without loading a model.
///
/// # Safety
/// Input bytes must be valid for their length. `output` must point to a writable handle slot.
#[no_mangle]
pub unsafe extern "C" fn sift_open(path: SiftString, output: *mut *mut Sift) -> i32 {
    boundary(|| {
        let output = unsafe { output.as_mut() }.context("The output handle pointer is null.")?;
        *output = std::ptr::null_mut();
        let path = unsafe { text(path, MAX_PATH_BYTES) }?;
        let engine = Engine::open(path)?;
        *output = Box::into_raw(Box::new(Sift {
            engine: Mutex::new(engine),
        }));
        Ok(())
    })
}

/// Create an index from a JSON document array and a local model directory.
///
/// # Safety
/// Input bytes must be valid for their lengths. `output` must point to a writable handle slot.
#[no_mangle]
pub unsafe extern "C" fn sift_create(
    path: SiftString,
    model: SiftString,
    documents: SiftString,
    output: *mut *mut Sift,
) -> i32 {
    boundary(|| {
        let output = unsafe { output.as_mut() }.context("The output handle pointer is null.")?;
        *output = std::ptr::null_mut();
        let path = unsafe { text(path, MAX_PATH_BYTES) }?;
        let model = unsafe { text(model, MAX_PATH_BYTES) }?;
        let documents: Vec<serde_json::Value> =
            serde_json::from_str(unsafe { text(documents, MAX_JSON_BYTES) }?)?;
        let engine = Engine::create(path, model, &documents)?;
        *output = Box::into_raw(Box::new(Sift {
            engine: Mutex::new(engine),
        }));
        Ok(())
    })
}

/// Search with the library's JSON request schema. Free the result with `sift_string_free`.
///
/// # Safety
/// The handle must be live. Input bytes must be valid. `output` must point to a writable string slot.
#[no_mangle]
pub unsafe extern "C" fn sift_search(
    handle: *mut Sift,
    request: SiftString,
    output: *mut *mut c_char,
) -> i32 {
    boundary(|| {
        let output = unsafe { output.as_mut() }.context("The output string pointer is null.")?;
        *output = std::ptr::null_mut();
        let options: SearchOptions =
            serde_json::from_str(unsafe { text(request, MAX_JSON_BYTES) }?)?;
        unsafe {
            with_engine(handle, |engine| {
                let result = serde_json::to_string(&engine.search(options)?)?;
                *output = CString::new(result)?.into_raw();
                Ok(())
            })
        }
    })
}

/// Write a JSON document array. Mode 0 inserts and mode 1 replaces older copies of each identifier.
///
/// # Safety
/// The handle must be live. Input bytes must be valid for their length.
#[no_mangle]
pub unsafe extern "C" fn sift_write(handle: *mut Sift, documents: SiftString, mode: i32) -> i32 {
    boundary(|| {
        let mode = match mode {
            0 => WriteMode::Insert,
            1 => WriteMode::Upsert,
            _ => return Err(anyhow!("Write mode must be 0 for insert or 1 for upsert.")),
        };
        let documents: Vec<serde_json::Value> =
            serde_json::from_str(unsafe { text(documents, MAX_JSON_BYTES) }?)?;
        unsafe {
            with_engine(handle, |engine| {
                engine.write_documents(&documents, mode).map(|_| ())
            })
        }
    })
}

/// Delete the string identifiers in a JSON array.
///
/// # Safety
/// The handle must be live. Input bytes must be valid for their length.
#[no_mangle]
pub unsafe extern "C" fn sift_delete(handle: *mut Sift, identifiers: SiftString) -> i32 {
    boundary(|| {
        let identifiers: Vec<String> =
            serde_json::from_str(unsafe { text(identifiers, MAX_JSON_BYTES) }?)?;
        unsafe { with_engine(handle, |engine| engine.delete(&identifiers).map(|_| ())) }
    })
}

/// Compact an index through the shared writer.
///
/// # Safety
/// The handle must be live for the complete call.
#[no_mangle]
pub unsafe extern "C" fn sift_compact(handle: *mut Sift) -> i32 {
    boundary(|| unsafe { with_engine(handle, Engine::compact) })
}

/// Reload the current index snapshot.
///
/// # Safety
/// The handle must be live for the complete call.
#[no_mangle]
pub unsafe extern "C" fn sift_reload(handle: *mut Sift) -> i32 {
    boundary(|| unsafe { with_engine(handle, Engine::reload) })
}

/// Close a handle. Null is accepted.
///
/// # Safety
/// Close each handle once. No operation may use the handle during or after this call.
#[no_mangle]
pub unsafe extern "C" fn sift_close(handle: *mut Sift) {
    if !handle.is_null() {
        let _ = boundary(|| {
            unsafe { drop(Box::from_raw(handle)) };
            Ok(())
        });
    }
}

/// Free an owned result string. Null is accepted.
///
/// # Safety
/// The pointer must be an unmodified result from this library. Free each result once.
#[no_mangle]
pub unsafe extern "C" fn sift_string_free(value: *mut c_char) {
    if !value.is_null() {
        unsafe { drop(CString::from_raw(value)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn rejects_null_oversized_and_invalid_utf8_inputs() {
        unsafe {
            assert!(text(
                SiftString {
                    data: std::ptr::null(),
                    len: 1
                },
                10
            )
            .is_err());
            assert!(text(
                SiftString {
                    data: std::ptr::null(),
                    len: 11
                },
                10
            )
            .is_err());
            let invalid = [0xff];
            assert!(text(
                SiftString {
                    data: invalid.as_ptr(),
                    len: 1
                },
                10
            )
            .is_err());
            assert_eq!(
                text(
                    SiftString {
                        data: std::ptr::null(),
                        len: 0
                    },
                    10
                )
                .unwrap(),
                ""
            );
        }
    }

    #[test]
    fn errors_stay_on_the_calling_thread_and_panics_do_not_cross_the_boundary() {
        assert_eq!(boundary(|| Err(anyhow!("Main thread error."))), 1);
        std::thread::spawn(|| {
            assert!(unsafe { CStr::from_ptr(sift_last_error()) }
                .to_bytes()
                .is_empty());
            assert_eq!(boundary(|| panic!("Injected internal failure.")), 2);
        })
        .join()
        .unwrap();
        assert_eq!(
            unsafe { CStr::from_ptr(sift_last_error()) }
                .to_str()
                .unwrap(),
            "Main thread error."
        );
        assert_eq!(boundary(|| Ok(())), 0);
        assert!(unsafe { CStr::from_ptr(sift_last_error()) }
            .to_bytes()
            .is_empty());
    }
}
