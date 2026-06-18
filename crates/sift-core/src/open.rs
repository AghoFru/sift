//! Artifact loading and mmap view construction.

use crate::{prewarm, static_bytes, static_cast, unpack_u24, DataView, Index, Meta};
use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;

impl Index {
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let meta: Meta = serde_json::from_str(
            &std::fs::read_to_string(dir.join("meta.json")).context("reading meta.json")?,
        )?;
        if meta.schema_version != 1 {
            return Err(anyhow!(
                "unsupported schema version {}",
                meta.schema_version
            ));
        }

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer: {e}"))?;

        let mm = |name: &str| -> Result<Mmap> {
            let p = dir.join(name);
            let f = File::open(&p).with_context(|| format!("opening {}", p.display()))?;
            let map = unsafe { Mmap::map(&f).with_context(|| format!("mmap {}", p.display()))? };
            // Hint the kernel that we'll need all of this; allows asynchronous
            // prefetch so cold queries pay less of the page-fault tax.
            let _ = map.advise(memmap2::Advice::WillNeed);
            Ok(map)
        };

        let mm_indptr = mm("indptr.bin")?;
        let mm_indices = mm("indices.bin")?;
        // Posting weights ship as either f16 (data_f16.bin, the build
        // default) or f32 (data.bin). Both are scored directly off the mmap;
        // f16 converts per access in the inner loop.
        let (mm_data, data_is_f16): (Mmap, bool) = match mm("data_f16.bin") {
            Ok(m) => (m, true),
            Err(_) => (
                mm("data.bin").context("neither data_f16.bin nor data.bin present in artifact")?,
                false,
            ),
        };
        // Exact CSR is optional (artifacts built before reranker support won't have it).
        let mm_exact_indptr = mm("exact_indptr.bin").ok();
        let mm_exact_indices = mm("exact_indices.bin").ok();
        let (mm_exact_data, exact_is_f16): (Option<Mmap>, bool) = match mm("exact_data_f16.bin") {
            Ok(m) => (Some(m), true),
            Err(_) => (mm("exact_data.bin").ok(), false),
        };
        // Bigram inverted index is also optional.
        let mm_bg_keys = mm("bigram_keys.bin").ok();
        let mm_bg_indptr = mm("bigram_indptr.bin").ok();
        let mm_bg_indices = mm("bigram_indices.bin").ok();
        let mm_bg_idf = mm("bigram_idf.bin").ok();
        let mm_spell_words = mm("spell_words.bin").ok();
        let mm_spell_word_offs = mm("spell_word_offs.bin").ok();
        let mm_spell_word_df = mm("spell_word_df.bin").ok();
        let mm_spell_del_keys = mm("spell_del_keys.bin").ok();
        let mm_spell_del_offs = mm("spell_del_offs.bin").ok();
        let mm_spell_del_ptr = mm("spell_del_ptr.bin").ok();
        let mm_spell_del_posts = mm("spell_del_posts.bin").ok();
        let mm_fwd_indptr = mm("fwd_indptr.bin").ok();
        let mm_fwd_indices = mm("fwd_indices.bin").ok();
        let mm_fwd_data = mm("fwd_data.bin").ok();
        let mm_pos_indptr = mm("pos_indptr.bin").ok();
        let mm_pos_terms = mm("pos_terms.bin").ok();
        let mm_qexp_indptr = mm("qexp_indptr.bin").ok();
        let mm_qexp_terms = mm("qexp_terms.bin").ok();
        let mm_qexp_sims = mm("qexp_sims.bin").ok();
        let mm_dedup = mm("dedup_canonical.bin").ok();
        let mm_block_max_indptr = mm("block_max_indptr.bin").ok();
        let mm_block_max = mm("block_max.bin").ok();

        // Custom ranking attribute sidecars, declared in rank_fields.json.
        let (rank_names, mut rank_mmaps): (Vec<String>, Vec<Mmap>) =
            match std::fs::read_to_string(dir.join("rank_fields.json")) {
                Ok(s) => {
                    let manifest: serde_json::Value =
                        serde_json::from_str(&s).with_context(|| "parsing rank_fields.json")?;
                    let names: Vec<String> = manifest
                        .get("fields")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();
                    let mut maps = Vec::with_capacity(names.len());
                    for n in &names {
                        let safe: String = n
                            .chars()
                            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                            .collect();
                        let m = mm(&format!("rank_{safe}.bin"))?;
                        maps.push(m);
                    }
                    (names, maps)
                }
                Err(_) => (Vec::new(), Vec::new()),
            };
        let mm_idf = mm("idf.bin")?;
        let mm_keep = mm("vocab_keep.bin")?;
        let mm_doc_lens = mm("doc_lens.bin")?;
        let mm_doc_ids_text = mm("doc_ids_text.bin")?;
        let mm_doc_ids_off = mm("doc_ids_off.bin")?;
        let mm_doc_snips_text = mm("doc_snips_text.bin")?;
        let mm_doc_snips_off = mm("doc_snips_off.bin")?;
        // Optional payload (stored source) sidecar.
        let mm_payload_text = mm("payload_text.bin").ok();
        let mm_payload_off = mm("payload_off.bin").ok();

        // SAFETY: extending lifetime to 'static is valid because the mmap fields
        // outlive the slice views; field-drop order ensures the slices are
        // released before the mmaps go away (in practice both are released at the
        // same drop site, but we never use the slices in Drop).
        let indptr: &'static [u64] = static_cast::<u64>(&mm_indptr)?;
        // indices.bin is either a raw u32 array (default) or a packed-u24
        // stream when meta.indices_packing == "u24". In the u24 case we
        // decode once into an owned buffer (kept in the struct, released
        // with the Index) and drop the mmap so we don't pay for both.
        let mut mm_indices_kept: Option<Mmap> = None;
        let mut owned_indices: Option<Box<[u32]>> = None;
        let indices: &'static [u32] = if meta.indices_packing == "u24" {
            let owned: Box<[u32]> = unpack_u24(&mm_indices)?.into_boxed_slice();
            drop(mm_indices); // release the packed mmap; the owned buffer is the truth now
                              // SAFETY: same rationale as static_cast - the boxed buffer is a
                              // field of the Index and outlives the slice view (the heap
                              // allocation is stable across the move into the struct).
            let slice = unsafe { std::mem::transmute::<&[u32], &'static [u32]>(&owned[..]) };
            owned_indices = Some(owned);
            slice
        } else {
            let slice: &'static [u32] = static_cast::<u32>(&mm_indices)?;
            mm_indices_kept = Some(mm_indices);
            slice
        };
        let data: DataView = if data_is_f16 {
            DataView::F16(static_cast::<half::f16>(&mm_data)?)
        } else {
            DataView::F32(static_cast::<f32>(&mm_data)?)
        };
        let idf: &'static [f32] = static_cast::<f32>(&mm_idf)?;
        let keep: &'static [u8] = static_bytes(&mm_keep);
        let doc_lens: &'static [f32] = static_cast::<f32>(&mm_doc_lens)?;
        let doc_ids_text: &'static [u8] = static_bytes(&mm_doc_ids_text);
        let doc_ids_off: &'static [u64] = static_cast::<u64>(&mm_doc_ids_off)?;
        let doc_snips_text: &'static [u8] = static_bytes(&mm_doc_snips_text);
        let doc_snips_off: &'static [u64] = static_cast::<u64>(&mm_doc_snips_off)?;
        let payload_text: Option<&'static [u8]> = mm_payload_text.as_ref().map(|m| static_bytes(m));
        let payload_off: Option<&'static [u64]> = mm_payload_off
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;

        let exact_indptr: Option<&'static [u64]> = mm_exact_indptr
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let exact_indices: Option<&'static [u32]> = mm_exact_indices
            .as_ref()
            .map(|m| static_cast::<u32>(m))
            .transpose()?;
        let exact_data: Option<DataView> = mm_exact_data
            .as_ref()
            .map(|m| -> Result<DataView> {
                Ok(if exact_is_f16 {
                    DataView::F16(static_cast::<half::f16>(m)?)
                } else {
                    DataView::F32(static_cast::<f32>(m)?)
                })
            })
            .transpose()?;
        let bigram_keys: Option<&'static [u64]> = mm_bg_keys
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let bigram_indptr: Option<&'static [u64]> = mm_bg_indptr
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let bigram_indices: Option<&'static [u32]> = mm_bg_indices
            .as_ref()
            .map(|m| static_cast::<u32>(m))
            .transpose()?;
        let bigram_idf: Option<&'static [f32]> = mm_bg_idf
            .as_ref()
            .map(|m| static_cast::<f32>(m))
            .transpose()?;
        let spell_words: Option<&'static [u8]> = mm_spell_words.as_ref().map(|m| static_bytes(m));
        let spell_word_offs: Option<&'static [u64]> = mm_spell_word_offs
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let spell_word_df: Option<&'static [u32]> = mm_spell_word_df
            .as_ref()
            .map(|m| static_cast::<u32>(m))
            .transpose()?;
        let spell_del_keys: Option<&'static [u8]> =
            mm_spell_del_keys.as_ref().map(|m| static_bytes(m));
        let spell_del_offs: Option<&'static [u64]> = mm_spell_del_offs
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let spell_del_ptr: Option<&'static [u64]> = mm_spell_del_ptr
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let spell_del_posts: Option<&'static [u32]> = mm_spell_del_posts
            .as_ref()
            .map(|m| static_cast::<u32>(m))
            .transpose()?;
        let fwd_indptr: Option<&'static [u64]> = mm_fwd_indptr
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let fwd_indices: Option<&'static [u32]> = mm_fwd_indices
            .as_ref()
            .map(|m| static_cast::<u32>(m))
            .transpose()?;
        let fwd_data: Option<&'static [f32]> = mm_fwd_data
            .as_ref()
            .map(|m| static_cast::<f32>(m))
            .transpose()?;
        let pos_indptr: Option<&'static [u64]> = mm_pos_indptr
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let pos_terms: Option<&'static [u32]> = mm_pos_terms
            .as_ref()
            .map(|m| static_cast::<u32>(m))
            .transpose()?;
        let qexp_indptr: Option<&'static [u64]> = mm_qexp_indptr
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let qexp_terms: Option<&'static [u32]> = mm_qexp_terms
            .as_ref()
            .map(|m| static_cast::<u32>(m))
            .transpose()?;
        let qexp_sims: Option<&'static [f32]> = mm_qexp_sims
            .as_ref()
            .map(|m| static_cast::<f32>(m))
            .transpose()?;
        let dedup_canonical: Option<&'static [u8]> = mm_dedup.as_ref().map(|m| static_bytes(m));
        let block_max_indptr: Option<&'static [u64]> = mm_block_max_indptr
            .as_ref()
            .map(|m| static_cast::<u64>(m))
            .transpose()?;
        let block_max: Option<&'static [f32]> = mm_block_max
            .as_ref()
            .map(|m| static_cast::<f32>(m))
            .transpose()?;

        let mut rank_attrs: std::collections::HashMap<String, &'static [f32]> =
            std::collections::HashMap::new();
        for (name, mmap) in rank_names.iter().zip(rank_mmaps.iter()) {
            let slice: &'static [f32] = static_cast::<f32>(mmap)?;
            if slice.len() != meta.n_docs as usize {
                return Err(anyhow!(
                    "rank attribute '{name}' has {} entries; expected {}",
                    slice.len(),
                    meta.n_docs
                ));
            }
            rank_attrs.insert(name.clone(), slice);
        }
        // we move rank_mmaps into the struct below; rebind for clarity
        let rank_mmaps = std::mem::take(&mut rank_mmaps);

        // Eagerly fault in the small always-hot files. These are linear in vocab
        // (~50 KB - 35 MB), so prewarm is sub-second even on the biggest index.
        // The huge indices.bin/data.bin files are left for the background warmup
        // pass in the serve binary to handle.
        prewarm(&mm_indptr);
        prewarm(&mm_idf);
        prewarm(&mm_keep);
        prewarm(&mm_doc_lens);
        prewarm(&mm_doc_ids_off);
        prewarm(&mm_doc_snips_off);

        // sanity checks
        let v = meta.vocab_size as usize;
        let n = meta.n_docs as usize;
        if indptr.len() != v + 1 {
            return Err(anyhow!("indptr len {} != V+1 {}", indptr.len(), v + 1));
        }
        if indices.len() as u64 != meta.n_nonzero {
            return Err(anyhow!(
                "indices len {} != nnz {}",
                indices.len(),
                meta.n_nonzero
            ));
        }
        if data.len() != indices.len() {
            return Err(anyhow!("data/indices len mismatch"));
        }
        if idf.len() != v {
            return Err(anyhow!("idf len {} != V {}", idf.len(), v));
        }
        if keep.len() != v {
            return Err(anyhow!("keep len {} != V {}", keep.len(), v));
        }
        if doc_lens.len() != n {
            return Err(anyhow!("doc_lens len {} != N {}", doc_lens.len(), n));
        }
        if doc_ids_off.len() != n + 1 || doc_snips_off.len() != n + 1 {
            return Err(anyhow!("doc_ids_off / doc_snips_off length mismatch"));
        }

        Ok(Index {
            meta,
            path: dir.to_path_buf(),
            tokenizer,
            _mm_indptr: mm_indptr,
            _mm_indices: mm_indices_kept,
            _owned_indices: owned_indices,
            _mm_data: mm_data,
            _mm_exact_indptr: mm_exact_indptr,
            _mm_exact_indices: mm_exact_indices,
            _mm_exact_data: mm_exact_data,
            _mm_bg_keys: mm_bg_keys,
            _mm_bg_indptr: mm_bg_indptr,
            _mm_bg_indices: mm_bg_indices,
            _mm_bg_idf: mm_bg_idf,
            _mm_spell_words: mm_spell_words,
            _mm_spell_word_offs: mm_spell_word_offs,
            _mm_spell_word_df: mm_spell_word_df,
            _mm_spell_del_keys: mm_spell_del_keys,
            _mm_spell_del_offs: mm_spell_del_offs,
            _mm_spell_del_ptr: mm_spell_del_ptr,
            _mm_spell_del_posts: mm_spell_del_posts,
            _mm_fwd_indptr: mm_fwd_indptr,
            _mm_fwd_indices: mm_fwd_indices,
            _mm_fwd_data: mm_fwd_data,
            _mm_pos_indptr: mm_pos_indptr,
            _mm_pos_terms: mm_pos_terms,
            _mm_qexp_indptr: mm_qexp_indptr,
            _mm_qexp_terms: mm_qexp_terms,
            _mm_qexp_sims: mm_qexp_sims,
            _mm_dedup: mm_dedup,
            _mm_block_max_indptr: mm_block_max_indptr,
            _mm_block_max: mm_block_max,
            _mm_rank: rank_mmaps,
            _mm_idf: mm_idf,
            _mm_keep: mm_keep,
            _mm_doc_lens: mm_doc_lens,
            _mm_doc_ids_text: mm_doc_ids_text,
            _mm_doc_ids_off: mm_doc_ids_off,
            _mm_doc_snips_text: mm_doc_snips_text,
            _mm_doc_snips_off: mm_doc_snips_off,
            _mm_payload_text: mm_payload_text,
            _mm_payload_off: mm_payload_off,
            indptr,
            indices,
            data,
            exact_indptr,
            exact_indices,
            exact_data,
            bigram_keys,
            bigram_indptr,
            bigram_indices,
            bigram_idf,
            spell_words,
            spell_word_offs,
            spell_word_df,
            spell_del_keys,
            spell_del_offs,
            spell_del_ptr,
            spell_del_posts,
            fwd_indptr,
            fwd_indices,
            fwd_data,
            pos_indptr,
            pos_terms,
            qexp_indptr,
            qexp_terms,
            qexp_sims,
            dedup_canonical,
            block_max_indptr,
            block_max,
            rank_attrs,
            idf,
            keep,
            doc_lens,
            doc_ids_text,
            doc_ids_off,
            doc_snips_text,
            doc_snips_off,
            payload_text,
            payload_off,
        })
    }
}
