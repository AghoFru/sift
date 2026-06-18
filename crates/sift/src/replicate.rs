//! `sift replicate` - consistent, incremental index replication.
//!
//! Replication leans entirely on the storage model: segment directories are
//! immutable once written, and `manifest.json` is the commit pointer. A
//! consistent replica is therefore produced by copying in the right order:
//!
//!   1. every segment the source manifest references that the destination is
//!      missing (immutable, so a name match is a content match: present
//!      segments are skipped),
//!   2. the tombstone set,
//!   3. the manifest LAST, written atomically (temp + fsync + rename + dir
//!      fsync), which is the instant the replica adopts the new generation.
//!
//! Because every referenced segment is on disk before the manifest naming it
//! is committed, the destination is never observed referencing a segment that
//! isn't fully present, even if the process dies mid-copy. Segments left over
//! from a previous generation (e.g. after the source compacted) are pruned
//! after the new manifest is adopted.
//!
//! `--from` / `--to` are filesystem paths: a local copy, or any mounted
//! filesystem (NFS/SMB/an rsync target). sift owns the consistent ordering;
//! the mount (or whatever you put underneath) owns the transport. This is the
//! commit-point file-copy scheme Solr's master/slave and Lucene's replicator
//! module formalize.

use anyhow::{anyhow, Context, Result};
use clap::Args;
use sift_core::{recover_orphans, Manifest};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct ReplicateArgs {
    /// Source to replicate FROM. Either a filesystem path (a manifested index
    /// or a single-artifact `sift build` output, local or on a mounted FS) or
    /// an `http(s)://host[:port]/[prefix/]replicate/<index>` URL served by a
    /// `sift serve --enable-replication` instance.
    #[arg(long)]
    pub from: String,
    /// Destination index directory to replicate INTO (created if absent).
    #[arg(long)]
    pub to: PathBuf,
    /// Poll and re-sync every N seconds. 0 (the default) does one sync and
    /// exits. With a positive value the command runs until interrupted,
    /// syncing only what changed each round.
    #[arg(long, default_value_t = 0)]
    pub watch: u64,
    /// Bearer token for an authenticated replication source (HTTP only).
    #[arg(long, env = "SIFT_REPLICATE_TOKEN")]
    pub token: Option<String>,
}

/// Outcome of one sync pass.
#[derive(Debug, Default)]
pub struct SyncReport {
    /// False when the destination was already at the source's generation.
    pub changed: bool,
    /// Source generation now on the destination.
    pub generation: u64,
    /// Segments newly copied this pass.
    pub segments_copied: usize,
    /// Segments the manifest references in total.
    pub segments_total: usize,
    /// Stale segment dirs pruned from the destination after adopting.
    pub pruned: usize,
}

pub fn run(args: ReplicateArgs) -> Result<()> {
    let is_http = args.from.starts_with("http://") || args.from.starts_with("https://");
    let do_sync = |to: &Path| -> Result<SyncReport> {
        if is_http {
            http::sync_http(&args.from, to, args.token.as_deref())
        } else {
            sync_once(Path::new(&args.from), to)
        }
    };
    if args.watch == 0 {
        let r = do_sync(&args.to)?;
        report(&args, &r);
        return Ok(());
    }
    println!(
        "sift replicate: watching {} -> {} every {}s",
        args.from,
        args.to.display(),
        args.watch
    );
    loop {
        match do_sync(&args.to) {
            Ok(r) => report(&args, &r),
            Err(e) => eprintln!("sift replicate: sync failed: {e:#}"),
        }
        std::thread::sleep(std::time::Duration::from_secs(args.watch));
    }
}

fn report(args: &ReplicateArgs, r: &SyncReport) {
    if r.changed {
        println!(
            "replicated {} -> {}: generation {}, {} segment(s) copied ({} total), {} pruned",
            args.from,
            args.to.display(),
            r.generation,
            r.segments_copied,
            r.segments_total,
            r.pruned
        );
    } else {
        println!(
            "{} already current at generation {}",
            args.to.display(),
            r.generation
        );
    }
}

/// Perform one consistent sync from `from` to `to`.
pub fn sync_once(from: &Path, to: &Path) -> Result<SyncReport> {
    if from.join("manifest.json").exists() {
        sync_manifested(from, to)
    } else if from.join("meta.json").exists() {
        sync_legacy(from, to)
    } else {
        Err(anyhow!(
            "{} is neither a multi-segment index (manifest.json) nor a .sift artifact (meta.json)",
            from.display()
        ))
    }
}

fn sync_manifested(from: &Path, to: &Path) -> Result<SyncReport> {
    let src = Manifest::read(from)?;
    fs::create_dir_all(to).with_context(|| format!("creating {}", to.display()))?;
    // Clear any temp dirs left by a previous interrupted run.
    clean_temp(to)?;

    // Short-circuit when the destination is already at this generation with
    // every referenced segment present.
    if to.join("manifest.json").exists() {
        if let Ok(dst) = Manifest::read(to) {
            if dst.generation == src.generation
                && src.segments.iter().all(|s| to.join(&s.dir).is_dir())
            {
                return Ok(SyncReport {
                    changed: false,
                    generation: src.generation,
                    segments_total: src.segments.len(),
                    ..Default::default()
                });
            }
        }
    }

    // 1. Copy missing segments (immutable: present ones are identical, skip).
    let mut copied = 0usize;
    for seg in &src.segments {
        let dst_seg = to.join(&seg.dir);
        if dst_seg.is_dir() {
            continue;
        }
        let src_seg = from.join(&seg.dir);
        if !src_seg.is_dir() {
            return Err(anyhow!(
                "source manifest references {} but it is missing on disk",
                src_seg.display()
            ));
        }
        let tmp = to.join(format!(".repl-{}.tmp", seg.dir));
        let _ = fs::remove_dir_all(&tmp);
        copy_tree(&src_seg, &tmp).with_context(|| format!("copying segment {}", seg.dir))?;
        sift_core::fsync_dir_contents(&tmp).ok();
        fs::rename(&tmp, &dst_seg).with_context(|| format!("publishing segment {}", seg.dir))?;
        sift_core::fsync_dir(to).ok();
        copied += 1;
    }

    // 2. Mirror the tombstone set (mutable: copy fresh, or drop if the source
    //    has none, e.g. right after a compaction).
    mirror_file(&from.join("tombstones"), &to.join("tombstones"))?;

    // 3. Adopt the new manifest last, durably. After this the replica is at
    //    the new generation and every referenced segment is already present.
    src.write(to).context("committing replicated manifest")?;

    // 4. Drop segments the new manifest no longer references (e.g. the inputs
    //    a source-side compaction replaced).
    let pruned = recover_orphans(to, &src).unwrap_or(0);

    Ok(SyncReport {
        changed: true,
        generation: src.generation,
        segments_copied: copied,
        segments_total: src.segments.len(),
        pruned,
    })
}

/// A legacy single-artifact directory (built offline by `sift build`, no
/// manifest). It is static, so a straight file copy is consistent; we still
/// copy the data files before `meta.json` (the file `Index::open` keys on) so
/// an interrupted copy never leaves a meta pointing at absent data.
fn sync_legacy(from: &Path, to: &Path) -> Result<SyncReport> {
    fs::create_dir_all(to).with_context(|| format!("creating {}", to.display()))?;
    let mut entries: Vec<PathBuf> = Vec::new();
    for e in fs::read_dir(from)? {
        let p = e?.path();
        if p.is_file() {
            entries.push(p);
        }
    }
    // meta.json last.
    entries.sort_by_key(|p| p.file_name().map(|n| n == "meta.json").unwrap_or(false));
    for src_file in &entries {
        let name = src_file.file_name().unwrap();
        copy_file_atomic(src_file, &to.join(name))?;
    }
    sift_core::fsync_dir(to).ok();
    Ok(SyncReport {
        changed: true,
        generation: 0,
        segments_copied: entries.len(),
        segments_total: entries.len(),
        pruned: 0,
    })
}

/// Copy `src` onto `dst` via a temp file + atomic rename, or remove `dst` when
/// `src` is absent (so the destination mirrors the source exactly).
fn mirror_file(src: &Path, dst: &Path) -> Result<()> {
    if src.exists() {
        copy_file_atomic(src, dst)
    } else {
        if dst.exists() {
            fs::remove_file(dst).with_context(|| format!("removing stale {}", dst.display()))?;
        }
        Ok(())
    }
}

/// Copy a single file to `dst` durably: stream into a sibling temp file, fsync
/// it, then rename into place.
fn copy_file_atomic(src: &Path, dst: &Path) -> Result<()> {
    let tmp = dst.with_extension("repl-tmp");
    {
        let mut r = fs::File::open(src).with_context(|| format!("opening {}", src.display()))?;
        let mut w =
            fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        std::io::copy(&mut r, &mut w).with_context(|| format!("copying {}", src.display()))?;
        w.sync_all().ok();
    }
    fs::rename(&tmp, dst).with_context(|| format!("renaming into {}", dst.display()))?;
    Ok(())
}

/// Recursively copy a directory tree to a fresh destination.
fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst).with_context(|| format!("creating {}", dst.display()))?;
    for entry in fs::read_dir(src).with_context(|| format!("reading {}", src.display()))? {
        let entry = entry?;
        let p = entry.path();
        let target = dst.join(entry.file_name());
        if p.is_dir() {
            copy_tree(&p, &target)?;
        } else {
            // Within a fresh temp tree a plain copy is fine; the directory is
            // fsync'd by the caller before it is published.
            let mut r = fs::File::open(&p).with_context(|| format!("opening {}", p.display()))?;
            let mut w = fs::File::create(&target)
                .with_context(|| format!("creating {}", target.display()))?;
            std::io::copy(&mut r, &mut w).with_context(|| format!("copying {}", p.display()))?;
        }
    }
    Ok(())
}

/// Remove leftover `.repl-*.tmp` staging dirs from an interrupted run.
fn clean_temp(dir: &Path) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(".repl-") && name.ends_with(".tmp") {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
    Ok(())
}

/// HTTP pull replication: same commit-point ordering as the filesystem path,
/// but segment files are streamed from a `sift serve --enable-replication`
/// source over `{from}/listing` + `{from}/file?path=<rel>`.
mod http {
    use super::SyncReport;
    use anyhow::{Context, Result};
    use serde::Deserialize;
    use sift_core::{recover_orphans, Manifest};
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::path::Path;

    #[derive(Deserialize)]
    struct FileEntry {
        path: String,
        len: u64,
    }
    #[derive(Deserialize)]
    struct Listing {
        generation: u64,
        layout: String,
        files: Vec<FileEntry>,
    }

    fn get(url: &str, token: Option<&str>) -> Result<ureq::Response> {
        let mut req = ureq::get(url);
        if let Some(t) = token {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        req.call().with_context(|| format!("GET {url}"))
    }

    /// Stream `{base}/file?path=<rel>` into `dest` atomically (temp + fsync +
    /// rename).
    fn download_file(base: &str, rel: &str, token: Option<&str>, dest: &Path) -> Result<()> {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let url = format!("{base}/file?path={}", urlencode(rel));
        let resp = get(&url, token)?;
        let tmp = dest.with_extension("repl-dl");
        {
            let mut reader = resp.into_reader();
            let mut f = std::fs::File::create(&tmp)
                .with_context(|| format!("creating {}", tmp.display()))?;
            std::io::copy(&mut reader, &mut f).with_context(|| format!("downloading {rel}"))?;
            f.flush().ok();
            f.sync_all().ok();
        }
        std::fs::rename(&tmp, dest).with_context(|| format!("renaming into {}", dest.display()))?;
        Ok(())
    }

    /// Minimal percent-encoding for a relative path used as a query value.
    fn urlencode(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'/' => {
                    out.push(b as char)
                }
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
        out
    }

    pub(super) fn sync_http(from: &str, to: &Path, token: Option<&str>) -> Result<SyncReport> {
        let base = from.trim_end_matches('/');
        let listing: Listing = get(&format!("{base}/listing"), token)?
            .into_json()
            .context("parsing replication listing")?;
        std::fs::create_dir_all(to).with_context(|| format!("creating {}", to.display()))?;
        super::clean_temp(to)?;

        let manifested = listing.layout == "manifest";
        let pointer = if manifested {
            "manifest.json"
        } else {
            "meta.json"
        };

        // Group files: segment dir -> its files (relative-within-segment),
        // plus loose root files (tombstones, legacy bins) and the pointer.
        let mut segments: BTreeMap<String, Vec<(String, u64)>> = BTreeMap::new();
        let mut root_files: Vec<(String, u64)> = Vec::new();
        for f in &listing.files {
            if f.path == pointer {
                continue;
            }
            match f.path.split_once('/') {
                Some((seg, rest)) if seg.starts_with("seg-") => {
                    segments
                        .entry(seg.to_string())
                        .or_default()
                        .push((rest.to_string(), f.len));
                }
                _ => root_files.push((f.path.clone(), f.len)),
            }
        }

        // No-op when already current.
        if manifested {
            if let Ok(local) = Manifest::read(to) {
                if local.generation == listing.generation
                    && segments.keys().all(|s| to.join(s).is_dir())
                {
                    return Ok(SyncReport {
                        changed: false,
                        generation: listing.generation,
                        segments_total: segments.len(),
                        ..Default::default()
                    });
                }
            }
        } else if to.join(pointer).exists()
            && listing.files.iter().all(|f| {
                std::fs::metadata(to.join(&f.path))
                    .map(|m| m.len() == f.len)
                    .unwrap_or(false)
            })
        {
            return Ok(SyncReport {
                changed: false,
                generation: 0,
                ..Default::default()
            });
        }

        // 1. Missing segments: download into a temp dir, fsync, rename. Present
        //    segment dirs are immutable, so skip them.
        let mut copied = 0usize;
        for (seg, files) in &segments {
            let final_dir = to.join(seg);
            if final_dir.is_dir() {
                continue;
            }
            let tmp = to.join(format!(".repl-{seg}.tmp"));
            let _ = std::fs::remove_dir_all(&tmp);
            std::fs::create_dir_all(&tmp)?;
            for (rel, _len) in files {
                download_file(base, &format!("{seg}/{rel}"), token, &tmp.join(rel))?;
            }
            sift_core::fsync_dir_contents(&tmp).ok();
            std::fs::rename(&tmp, &final_dir)
                .with_context(|| format!("publishing segment {seg}"))?;
            sift_core::fsync_dir(to).ok();
            copied += 1;
        }

        // 2. Loose root files (tombstones, or a legacy artifact's bins).
        for (rel, _len) in &root_files {
            download_file(base, rel, token, &to.join(rel))?;
        }

        // 3. Pointer last (atomic adopt).
        download_file(base, pointer, token, &to.join(pointer))?;
        sift_core::fsync_dir(to).ok();

        // 4. Prune segments the new manifest no longer references.
        let pruned = if manifested {
            Manifest::read(to)
                .ok()
                .map(|m| recover_orphans(to, &m).unwrap_or(0))
                .unwrap_or(0)
        } else {
            0
        };

        Ok(SyncReport {
            changed: true,
            generation: listing.generation,
            segments_copied: copied,
            segments_total: segments.len(),
            pruned,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sift_core::SegmentRef;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sift_repl_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_seg(dir: &Path, name: &str, body: &str) {
        let seg = dir.join(name);
        fs::create_dir_all(&seg).unwrap();
        fs::write(seg.join("data.bin"), body).unwrap();
    }

    fn manifest_with(dir: &Path, generation: u64, segs: &[&str]) {
        let m = Manifest {
            format: sift_core::MANIFEST_FORMAT,
            generation,
            segments: segs
                .iter()
                .map(|s| SegmentRef {
                    dir: s.to_string(),
                    source: None,
                    build_args: vec![],
                })
                .collect(),
        };
        m.write(dir).unwrap();
    }

    #[test]
    fn replicates_then_incrementally_syncs_and_prunes() {
        let src = scratch("src");
        let dst = scratch("dst");

        // Generation 1: two segments.
        write_seg(&src, "seg-00000", "a");
        write_seg(&src, "seg-00001", "b");
        manifest_with(&src, 1, &["seg-00000", "seg-00001"]);

        let r = sync_once(&src, &dst).unwrap();
        assert!(r.changed && r.segments_copied == 2 && r.generation == 1);
        assert_eq!(
            fs::read_to_string(dst.join("seg-00000/data.bin")).unwrap(),
            "a"
        );
        assert_eq!(
            fs::read_to_string(dst.join("seg-00001/data.bin")).unwrap(),
            "b"
        );

        // No-op when already current.
        let r = sync_once(&src, &dst).unwrap();
        assert!(!r.changed && r.generation == 1);

        // Generation 2: add a segment. Only the new one copies.
        write_seg(&src, "seg-00002", "c");
        manifest_with(&src, 2, &["seg-00000", "seg-00001", "seg-00002"]);
        let r = sync_once(&src, &dst).unwrap();
        assert!(r.changed && r.segments_copied == 1 && r.generation == 2);

        // Generation 3: a compaction replaces everything with one segment.
        // The replica copies it and prunes the three stale ones.
        write_seg(&src, "seg-00003", "merged");
        // remove old src segments to mimic compaction cleanup
        for s in ["seg-00000", "seg-00001", "seg-00002"] {
            fs::remove_dir_all(src.join(s)).unwrap();
        }
        manifest_with(&src, 3, &["seg-00003"]);
        let r = sync_once(&src, &dst).unwrap();
        assert!(r.changed && r.segments_copied == 1 && r.pruned == 3);
        assert!(dst.join("seg-00003").is_dir());
        assert!(!dst.join("seg-00000").exists());
        assert!(!dst.join("seg-00002").exists());
        assert_eq!(
            fs::read_to_string(dst.join("seg-00003/data.bin")).unwrap(),
            "merged"
        );

        fs::remove_dir_all(&src).ok();
        fs::remove_dir_all(&dst).ok();
    }

    #[test]
    fn mirrors_tombstones() {
        let src = scratch("ts_src");
        let dst = scratch("ts_dst");
        write_seg(&src, "seg-00000", "a");
        manifest_with(&src, 1, &["seg-00000"]);
        fs::write(src.join("tombstones"), "x\n").unwrap();
        sync_once(&src, &dst).unwrap();
        assert_eq!(fs::read_to_string(dst.join("tombstones")).unwrap(), "x\n");

        // Source drops the tombstone file (post-compaction); replica mirrors it.
        fs::remove_file(src.join("tombstones")).unwrap();
        manifest_with(&src, 2, &["seg-00000"]);
        sync_once(&src, &dst).unwrap();
        assert!(!dst.join("tombstones").exists());

        fs::remove_dir_all(&src).ok();
        fs::remove_dir_all(&dst).ok();
    }
}
