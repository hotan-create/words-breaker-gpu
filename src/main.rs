use anyhow::{Context, Result};
use bip39::{Language, Mnemonic};
use bitcoin::address::{Address, NetworkChecked, NetworkUnchecked};
use bitcoin::bip32::{DerivationPath, Xpriv};
use bitcoin::secp256k1::Secp256k1;
use bitcoin::{Network, PublicKey};
use clap::Parser;
use itertools::Itertools;
use rayon::prelude::*;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

mod candidates;
mod gpu;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    about = "Try permutations of BIP-39 words / token slots to match a BTC legacy address.",
    version
)]
struct Args {
    /// Target legacy Bitcoin address (Base58, starts with '1').
    target_address: Option<String>,

    /// BIP-39 wordlist language
    #[arg(long, short, default_value = "english")]
    language: String,

    /// Path to a tokenlist file.
    ///
    /// FILE FORMAT
    /// ===========
    /// • One line = one SLOT (= one token).
    /// • Blank lines and lines starting with '#' are ignored.
    /// • Within a line, ALTERNATIVES are separated by whitespace.
    /// • Within an alternative, WORDS are separated by commas.
    /// • A bare '?' inside an alternative marks a MISSING word to brute-force.
    ///
    /// EXAMPLE
    ///   zebra,liquid,tornado,?   abandon,art   <- slot 1
    ///   orbit,galaxy                           <- slot 2
    ///   venture,sun                            <- slot 3
    #[arg(long, value_name = "FILE")]
    tokenlist: Option<PathBuf>,

    /// Keep SLOT order as written in the file (do not permute slots).
    #[arg(long)]
    keep_token_order: bool,

    /// Keep WORD order within each slot as written.
    #[arg(long)]
    keep_word_order: bool,

    /// Minimum number of slots to use (default: all slots).
    #[arg(long, value_name = "N")]
    min_token: Option<usize>,

    /// Number of CPU threads (0 = all cores).
    #[arg(long, short, default_value_t = 0)]
    threads: usize,

    /// Verify GPU crypto primitives against CPU reference then exit.
    #[arg(long)]
    selftest: bool,

    /// Force CPU search (skip GPU even if available).
    #[arg(long)]
    cpu: bool,

    /// Override GPU batch size exactly (default: auto-probe).
    /// Example: --batch-size 65536
    #[arg(long, value_name = "N")]
    batch_size: Option<usize>,

    /// Probe START: batch size = 2^EXP (default: 16 = 65 536).
    /// Probe begins here and doubles until --max-batch or memory error.
    /// Example: --min-batch 16
    #[arg(long, value_name = "EXP", default_value_t = 16)]
    min_batch: u32,

    /// Probe CAP: batch size never exceeds 2^EXP (default: 28 = 268M).
    /// Set to 16 to cap at 65 536 — safe for 2 GB VRAM.
    /// Example: --max-batch 16
    #[arg(long, value_name = "EXP", default_value_t = 28)]
    max_batch: u32,
}


// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Token {
    Word(String),
    Missing,
}

type Alternative = Vec<Token>;
type Slot = Vec<Alternative>;

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    if args.selftest {
        println!("Running GPU primitive selftests...");
        let ok = gpu::run_selftest()?;
        if ok {
            println!("All selftests passed.");
            // WSL2: skip cuCtxDestroy crash
            std::process::exit(0);
        } else {
            eprintln!("One or more selftests FAILED");
            std::process::exit(1);
        }
    }

    let target_address = args
        .target_address
        .as_deref()
        .context("Missing target address")?;

    let target: Address<NetworkChecked> = target_address
        .parse::<Address<NetworkUnchecked>>()
        .context("Invalid Bitcoin address")?
        .require_network(Network::Bitcoin.into())
        .context("Only mainnet legacy addresses are supported")?;

    let language = parse_language(&args.language)?;
    let start = Instant::now();

    let tokenlist_path = args
        .tokenlist
        .as_ref()
        .context("--tokenlist is required")?;

    let slots = parse_tokenlist(tokenlist_path)?;
    validate_slots(&slots, language)?;

    let found = if args.cpu {
        println!("--cpu flag set: using CPU.");
        run_search_cpu(&args, &slots, &target, language)?
    } else {
        match run_search_gpu(&args, &slots, &target, language) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("GPU unavailable ({e:#}); falling back to CPU.");
                run_search_cpu(&args, &slots, &target, language)?
            }
        }
    };

    let elapsed = start.elapsed();
    if !found {
        println!("Exhausted all candidates without a match (elapsed: {elapsed:?})");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tokenlist parsing
// ---------------------------------------------------------------------------

fn parse_tokenlist(path: &PathBuf) -> Result<Vec<Slot>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read tokenlist: {}", path.display()))?;

    let mut slots: Vec<Slot> = Vec::new();

    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let alternatives: Vec<Alternative> = line
            .split_whitespace()
            .map(|alt_str| {
                alt_str
                    .split(',')
                    .filter(|t| !t.is_empty())
                    .map(|t| {
                        let t = t.trim();
                        if t == "?" {
                            Token::Missing
                        } else {
                            Token::Word(t.to_string())
                        }
                    })
                    .collect::<Alternative>()
            })
            .filter(|alt| !alt.is_empty())
            .collect();

        if alternatives.is_empty() {
            eprintln!("Warning: line {} empty after parsing, skipping.", lineno + 1);
            continue;
        }

        slots.push(alternatives);
    }

    if slots.is_empty() {
        anyhow::bail!("Tokenlist is empty or has no valid lines");
    }

    println!("Loaded {} slot(s) from tokenlist.", slots.len());
    Ok(slots)
}

fn validate_slots(slots: &[Slot], language: Language) -> Result<()> {
    let wordlist: &'static [&'static str] = language.words_by_prefix("");
    for (si, slot) in slots.iter().enumerate() {
        for (ai, alt) in slot.iter().enumerate() {
            for token in alt {
                if let Token::Word(w) = token {
                    if !wordlist.contains(&w.as_str()) {
                        anyhow::bail!(
                            "Slot {}, alternative {}: '{}' is not in the BIP-39 wordlist",
                            si + 1, ai + 1, w
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Token / alternative expansion helpers
// ---------------------------------------------------------------------------

fn expand_alternative(alt: &Alternative) -> (Vec<String>, Vec<usize>) {
    let mut words: Vec<String> = Vec::new();
    let mut missing_positions: Vec<usize> = Vec::new();
    for (i, token) in alt.iter().enumerate() {
        match token {
            Token::Word(w) => words.push(w.clone()),
            Token::Missing => missing_positions.push(i),
        }
    }
    (words, missing_positions)
}

fn slot_candidates(
    known_indices: &[u16],
    missing_positions: &[usize],
    total_len: usize,
    keep_word_order: bool,
    wordlist_len: usize,
) -> Vec<Vec<u16>> {
    let missing_count = missing_positions.len();

    let known_perms: Box<dyn Iterator<Item = Vec<u16>>> = if keep_word_order {
        Box::new(std::iter::once(known_indices.to_vec()))
    } else {
        let n = known_indices.len();
        Box::new(known_indices.iter().copied().permutations(n))
    };

    let mut results: Vec<Vec<u16>> = Vec::new();

    for known_perm in known_perms {
        for missing_values in (0..wordlist_len as u16).permutations_with_replacement(missing_count) {
            let mut seq: Vec<u16> = Vec::with_capacity(total_len);
            let mut ki = 0usize;
            let mut mi = 0usize;
            for pos in 0..total_len {
                if missing_positions.contains(&pos) {
                    seq.push(missing_values[mi]);
                    mi += 1;
                } else {
                    seq.push(known_perm[ki]);
                    ki += 1;
                }
            }
            results.push(seq);
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Lazy phrase iterator — O(1) RAM, streams to GPU without Vec allocation
// ---------------------------------------------------------------------------

/// Iterates over all 12-word phrase combinations without collecting into a Vec.
/// This is the key fix for OOM: 1B candidates × 24 bytes = 24 GB if collected.
/// With LazyPhraseIter, only one [u16;12] lives in memory at a time.
struct LazyPhraseIter {
    // [slot_idx][alt_idx][cand_idx] = Vec<u16> (partial phrase for that slot+alt)
    slot_alts: Vec<Vec<Vec<Vec<u16>>>>,
    // Current cursor: which alt and which cand within that alt, per slot
    alt_idx: Vec<usize>,
    cand_idx: Vec<usize>,
    // Slot orderings to try (permutations of slot indices, or just one if keep_token_order)
    slot_orders: Vec<Vec<usize>>,
    order_pos: usize, // which ordering we're on
    done: bool,
    first: bool,
}

impl LazyPhraseIter {
    fn new(
        chosen_slots: &[&Slot],
        keep_token_order: bool,
        keep_word_order: bool,
        wordlist: &'static [&'static str],
    ) -> Self {
        let n = chosen_slots.len();
        let wordlist_len = wordlist.len();

        // Pre-expand each slot × alt into concrete partial-phrase Vec<u16>
        // This is small: at most a few thousand entries per slot
        let slot_alts: Vec<Vec<Vec<Vec<u16>>>> = chosen_slots
            .iter()
            .map(|slot| {
                slot.iter()
                    .map(|alt| {
                        let (known_words, missing_positions) = expand_alternative(alt);
                        let total_len = alt.len();
                        let known_indices: Vec<u16> = known_words
                            .iter()
                            .map(|w| {
                                wordlist
                                    .iter()
                                    .position(|x| *x == w.as_str())
                                    .unwrap() as u16
                            })
                            .collect();
                        slot_candidates(
                            &known_indices,
                            &missing_positions,
                            total_len,
                            keep_word_order,
                            wordlist_len,
                        )
                    })
                    .collect()
            })
            .collect();

        let slot_orders: Vec<Vec<usize>> = if keep_token_order || n <= 1 {
            vec![(0..n).collect()]
        } else {
            (0..n).permutations(n).collect()
        };

        let num_slots = slot_alts.len();
        LazyPhraseIter {
            slot_alts,
            alt_idx: vec![0; num_slots],
            cand_idx: vec![0; num_slots],
            slot_orders,
            order_pos: 0,
            done: num_slots == 0,
            first: true,
        }
    }

    /// Advance cursor for the given slot ordering.
    /// Returns true if successfully advanced, false if this ordering is exhausted.
    fn advance_for_order(&mut self, order: &[usize]) -> bool {
        let n = order.len();
        let mut pos = n as isize - 1;
        while pos >= 0 {
            let si = order[pos as usize];
            let ai = self.alt_idx[si];
            self.cand_idx[si] += 1;
            if self.cand_idx[si] < self.slot_alts[si][ai].len() {
                return true;
            }
            // exhausted candidates in this alt — try next alt
            self.cand_idx[si] = 0;
            self.alt_idx[si] += 1;
            if self.alt_idx[si] < self.slot_alts[si].len() {
                return true;
            }
            // exhausted alts for this slot — reset and carry
            self.alt_idx[si] = 0;
            self.cand_idx[si] = 0;
            pos -= 1;
        }
        false
    }

    /// Build the current phrase for the given slot ordering.
    /// Returns None if the total word count != 12.
    fn build_phrase(&self, order: &[usize]) -> Option<[u16; 12]> {
        let mut phrase = [0u16; 12];
        let mut offset = 0usize;
        for &si in order {
            let ai = self.alt_idx[si];
            let ci = self.cand_idx[si];
            let words = &self.slot_alts[si][ai][ci];
            if offset + words.len() > 12 {
                return None;
            }
            phrase[offset..offset + words.len()].copy_from_slice(words);
            offset += words.len();
        }
        if offset == 12 { Some(phrase) } else { None }
    }
}

impl Iterator for LazyPhraseIter {
    type Item = [u16; 12];

    fn next(&mut self) -> Option<[u16; 12]> {
        if self.done {
            return None;
        }

        loop {
            if self.order_pos >= self.slot_orders.len() {
                self.done = true;
                return None;
            }

            let order = self.slot_orders[self.order_pos].clone();

            if self.first {
                self.first = false;
                if let Some(p) = self.build_phrase(&order) {
                    return Some(p);
                }
                // phrase invalid (!=12 words) — fall through to advance
            }

            // Try to advance
            if self.advance_for_order(&order) {
                if let Some(p) = self.build_phrase(&order) {
                    return Some(p);
                }
                // phrase invalid — keep advancing
                continue;
            }

            // This ordering exhausted — move to next ordering, reset cursors
            self.order_pos += 1;
            self.first = true;
            for v in self.alt_idx.iter_mut() {
                *v = 0;
            }
            for v in self.cand_idx.iter_mut() {
                *v = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU search — lazy streaming, auto batch size, no OOM
// ---------------------------------------------------------------------------

fn run_search_gpu(
    args: &Args,
    slots: &[Slot],
    target: &Address<NetworkChecked>,
    language: Language,
) -> Result<bool> {
    let gpu_handle = gpu::Gpu::new()?;
    let wordlist: &'static [&'static str] = language.words_by_prefix("");
    let gpu_wordlist = gpu::GpuWordlist::new(wordlist)?;
    let target_h160 = p2pkh_hash160(target)?;

    // Batch size: CLI override, atau adaptive probe mulai dari 1<<16.
    let batch_size = if let Some(b) = args.batch_size {
        println!("Using GPU (CUDA) — batch_size: {} (manual override)", format_number(b));
        b
    } else {
        probe_batch_size(&gpu_handle, &gpu_wordlist, &target_h160, args.min_batch, args.max_batch)
    };

    let min_token = args.min_token.unwrap_or(slots.len()).min(slots.len());
    let max_token = slots.len();
    println!("Trying slot subsets {min_token}..={max_token}.");

    let mut global_checked: usize = 0;

    for slot_count in min_token..=max_token {
        let slot_indices: Vec<usize> = (0..slots.len()).collect();

        for chosen_indices in slot_indices.iter().copied().combinations(slot_count) {
            let chosen: Vec<&Slot> = chosen_indices.iter().map(|&i| &slots[i]).collect();

            println!("Slot combination {:?}: streaming to GPU...", chosen_indices);
            let _ = io::stdout().flush();

            let iter = LazyPhraseIter::new(
                &chosen,
                args.keep_token_order,
                args.keep_word_order,
                wordlist,
            );

            match gpu_handle.search(iter, &gpu_wordlist, &target_h160, batch_size)? {
                Some(hit) => {
                    let phrase: Vec<&str> =
                        hit.indices.iter().map(|&i| wordlist[i as usize]).collect();
                    println!("Found matching mnemonic: {}", phrase.join(" "));
                    println!(
                        "Candidate index (0-based): {} (global: {})",
                        hit.global_index,
                        global_checked + hit.global_index,
                    );
                    println!("Derived address: {target}");
                    let _ = io::stdout().flush();
                    std::mem::forget(gpu_handle);
                    std::process::exit(0);
                }
                None => {
                    println!(
                        "Combination {:?}: no match. Global checked: ~{}",
                        chosen_indices,
                        format_number(global_checked),
                    );
                    global_checked += 1;
                }
            }
        }
    }

    std::mem::forget(gpu_handle);
    Ok(false)
}

/// Adaptive batch size probe.
///
/// Mulai dari START (64K), jalankan batch dummy.
/// Sukses → naikkan 2x. Gagal (OOM/error) → pakai yang terakhir sukses.
/// Berhenti jika batch sudah >500 MB (transfer overhead mulai dominan).
fn probe_batch_size(
    gpu: &gpu::Gpu,
    wordlist: &gpu::GpuWordlist,
    target_h160: &[u8; 20],
    min_exp: u32,  // start = 2^min_exp  (e.g. 16 = 65 536)
    max_exp: u32,  // cap   = 2^max_exp  (e.g. 28 = 268M, or 16 to hard-cap at 65K)
) -> usize {
    let start: usize = 1usize.checked_shl(min_exp).unwrap_or(1 << 16);
    let max:   usize = 1usize.checked_shl(max_exp).unwrap_or(1 << 28);

    fn make_dummy(n: usize) -> impl Iterator<Item = [u16; 12]> {
        (0..n).map(|_| [0u16; 12])
    }

    let mut batch   = start;
    let mut last_ok = start;

    println!(
        "Probing GPU batch size: 2^{min_exp}={} .. 2^{max_exp}={}",
        format_number(start),
        format_number(max),
    );

    loop {
        let t0      = std::time::Instant::now();
        let result  = gpu.search(make_dummy(batch), wordlist, target_h160, batch);
        let elapsed = t0.elapsed();

        match result {
            Ok(_) => {
                let throughput = batch as f64 / elapsed.as_secs_f64();
                println!(
                    "  batch {:>10} → OK  ({:.1} MB, {}/s, {:.0}ms)",
                    format_number(batch),
                    (batch * BYTES_PER_CAND_TOTAL) as f64 / (1024.0 * 1024.0),
                    format_number(throughput as usize),
                    elapsed.as_millis(),
                );
                last_ok = batch;

                let next = batch.saturating_mul(2);
                if next > max {
                    println!("  → reached --max-batch cap (2^{max_exp}={}).", format_number(max));
                    break;
                }
                batch = next;
            }
            Err(e) => {
                println!(
                    "  batch {:>10} → GAGAL ({:#}), pakai: {}",
                    format_number(batch),
                    e,
                    format_number(last_ok),
                );
                break;
            }
        }
    }

    let final_mb = (last_ok * BYTES_PER_CAND_TOTAL) as f64 / (1024.0 * 1024.0);
    println!(
        "Using GPU (CUDA) — batch_size: {} ({final_mb:.1} MB/batch)",
        format_number(last_ok),
    );
    let _ = io::stdout().flush();
    last_ok
}

// ---------------------------------------------------------------------------
// Batch size constants
// ---------------------------------------------------------------------------

/// VRAM bytes consumed per candidate across all active DeviceBuffers in gpu.search():
///   d_cand:      12 × u16  = 24 bytes
///   d_survivors:  1 × u32  =  4 bytes
const BYTES_PER_CAND_TOTAL: usize = 28;


// ---------------------------------------------------------------------------
// CPU search (unchanged from original, kept for --cpu fallback)
// ---------------------------------------------------------------------------

fn run_search_cpu(
    args: &Args,
    slots: &[Slot],
    target: &Address<NetworkChecked>,
    language: Language,
) -> Result<bool> {
    let num_threads = if args.threads == 0 {
        num_cpus::get()
    } else {
        args.threads
    };
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();
    println!("Using CPU with {num_threads} threads.");

    let wordlist: &'static [&'static str] = language.words_by_prefix("");
    let target_str = target.to_string();
    let derivation_path: DerivationPath = "m/44'/0'/0'/0/0".parse()?;
    let secp = Arc::new(Secp256k1::new());

    let min_token = args.min_token.unwrap_or(slots.len()).min(slots.len());
    let max_token = slots.len();
    println!("Trying slot subsets {min_token}..={max_token}.");

    let counter = Arc::new(AtomicUsize::new(0));
    let found = Arc::new(AtomicBool::new(false));
    let found_phrase = Arc::new(std::sync::Mutex::new(String::new()));
    let found_index = Arc::new(AtomicUsize::new(0));

    'outer: for slot_count in min_token..=max_token {
        if found.load(Ordering::Relaxed) {
            break;
        }

        let slot_indices: Vec<usize> = (0..slots.len()).collect();

        for chosen_indices in slot_indices.iter().copied().combinations(slot_count) {
            if found.load(Ordering::Relaxed) {
                break 'outer;
            }

            let chosen: Vec<&Slot> = chosen_indices.iter().map(|&i| &slots[i]).collect();

            // CPU path: collect to Vec is OK here since Rayon parallelises over it.
            // For very large sets (>100M) consider switching to LazyPhraseIter here too.
            let phrases: Vec<Vec<u16>> = enumerate_phrases(
                &chosen,
                args.keep_token_order,
                args.keep_word_order,
                wordlist,
            )
            .into_iter()
            .filter(|p| p.len() == 12)
            .collect();

            println!(
                "Slot combination {:?}: {} phrase(s) to check on CPU.",
                chosen_indices,
                format_number(phrases.len()),
            );
            let _ = io::stdout().flush();

            phrases.into_par_iter().for_each(|phrase_indices: Vec<u16>| {
                if found.load(Ordering::Relaxed) {
                    return;
                }

                let i = counter.fetch_add(1, Ordering::Relaxed);
                if i % 100_000 == 0 && i > 0 {
                    println!("Checked {} candidates...", format_number(i));
                    let _ = io::stdout().flush();
                }

                let phrase: Vec<&str> =
                    phrase_indices.iter().map(|&idx| wordlist[idx as usize]).collect();
                let phrase_str = phrase.join(" ");

                let mnemonic =
                    match Mnemonic::parse_in_normalized(language, &phrase_str) {
                        Ok(m) => m,
                        Err(_) => return,
                    };

                let seed = mnemonic.to_seed("");
                let master_xprv = match Xpriv::new_master(Network::Bitcoin, &seed) {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let child_xprv =
                    match master_xprv.derive_priv(&secp, &derivation_path) {
                        Ok(x) => x,
                        Err(_) => return,
                    };
                let child_pub =
                    PublicKey::new(child_xprv.private_key.public_key(&secp));
                let addr: Address<NetworkChecked> =
                    Address::p2pkh(&child_pub, Network::Bitcoin);

                if addr.to_string() == target_str {
                    found.store(true, Ordering::SeqCst);
                    found_index.store(i, Ordering::SeqCst);
                    *found_phrase.lock().unwrap() = phrase_str;
                }
            });

            if found.load(Ordering::SeqCst) {
                break 'outer;
            }
        }
    }

    if found.load(Ordering::SeqCst) {
        let fp = found_phrase.lock().unwrap();
        let idx = found_index.load(Ordering::SeqCst);
        println!("Found matching mnemonic: {fp}");
        println!("Candidate index (0-based): {idx}");
        println!("Derived address: {target_str}");
        Ok(true)
    } else {
        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// enumerate_phrases — dipakai oleh CPU path
// (GPU path pakai LazyPhraseIter di atas)
// ---------------------------------------------------------------------------

fn enumerate_phrases(
    chosen_slots: &[&Slot],
    keep_token_order: bool,
    keep_word_order: bool,
    wordlist: &'static [&'static str],
) -> Vec<Vec<u16>> {
    let wordlist_len = wordlist.len();

    let slot_alt_candidates: Vec<Vec<Vec<Vec<u16>>>> = chosen_slots
        .iter()
        .map(|slot| {
            slot.iter()
                .map(|alt| {
                    let (known_words, missing_positions) = expand_alternative(alt);
                    let total_len = alt.len();
                    let known_indices: Vec<u16> = known_words
                        .iter()
                        .map(|w| {
                            wordlist
                                .iter()
                                .position(|x| *x == w.as_str())
                                .unwrap() as u16
                        })
                        .collect();
                    slot_candidates(
                        &known_indices,
                        &missing_positions,
                        total_len,
                        keep_word_order,
                        wordlist_len,
                    )
                })
                .collect()
        })
        .collect();

    let num_slots = chosen_slots.len();
    let slot_orderings: Vec<Vec<usize>> = if keep_token_order {
        vec![(0..num_slots).collect()]
    } else {
        (0..num_slots).permutations(num_slots).collect()
    };

    let mut results: Vec<Vec<u16>> = Vec::new();

    for order in &slot_orderings {
        let alt_counts: Vec<usize> =
            order.iter().map(|&si| slot_alt_candidates[si].len()).collect();

        for alt_indices in alt_counts.iter().map(|&n| 0..n).multi_cartesian_product() {
            let per_slot: Vec<&Vec<Vec<u16>>> = order
                .iter()
                .zip(alt_indices.iter())
                .map(|(&si, &ai)| &slot_alt_candidates[si][ai])
                .collect();

            let cand_counts: Vec<usize> = per_slot.iter().map(|c| c.len()).collect();
            for cand_indices in
                cand_counts.iter().map(|&n| 0..n).multi_cartesian_product()
            {
                let phrase: Vec<u16> = per_slot
                    .iter()
                    .zip(cand_indices.iter())
                    .flat_map(|(cands, &ci)| cands[ci].iter().copied())
                    .collect();
                results.push(phrase);
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Permutations-with-replacement helper
// ---------------------------------------------------------------------------

trait PermutationsWithReplacement: Iterator + Sized {
    fn permutations_with_replacement(self, k: usize) -> PermWithReplacement<Self::Item>
    where
        Self::Item: Clone;
}

struct PermWithReplacement<T> {
    pool: Vec<T>,
    indices: Vec<usize>,
    k: usize,
    first: bool,
    done: bool,
}

impl<I: Iterator> PermutationsWithReplacement for I
where
    I::Item: Clone,
{
    fn permutations_with_replacement(self, k: usize) -> PermWithReplacement<I::Item> {
        let pool: Vec<I::Item> = self.collect();
        let n = pool.len();
        if k == 0 || n == 0 {
            return PermWithReplacement {
                pool,
                indices: vec![],
                k,
                first: true,
                done: k != 0,
            };
        }
        PermWithReplacement {
            pool,
            indices: vec![0usize; k],
            k,
            first: true,
            done: false,
        }
    }
}

impl<T: Clone> Iterator for PermWithReplacement<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        if self.done {
            return None;
        }
        if self.k == 0 {
            self.done = true;
            return Some(vec![]);
        }
        if self.first {
            self.first = false;
            return Some(
                self.indices.iter().map(|&i| self.pool[i].clone()).collect(),
            );
        }
        let n = self.pool.len();
        let mut pos = self.k as isize - 1;
        while pos >= 0 && self.indices[pos as usize] == n - 1 {
            self.indices[pos as usize] = 0;
            pos -= 1;
        }
        if pos < 0 {
            self.done = true;
            return None;
        }
        self.indices[pos as usize] += 1;
        Some(self.indices.iter().map(|&i| self.pool[i].clone()).collect())
    }
}

// ---------------------------------------------------------------------------
// P2PKH helpers
// ---------------------------------------------------------------------------

fn p2pkh_hash160(addr: &Address<NetworkChecked>) -> Result<[u8; 20]> {
    let spk = addr.script_pubkey();
    let bytes = spk.as_bytes();
    if bytes.len() == 25
        && bytes[0] == 0x76
        && bytes[1] == 0xa9
        && bytes[2] == 0x14
    {
        let mut h = [0u8; 20];
        h.copy_from_slice(&bytes[3..23]);
        Ok(h)
    } else {
        anyhow::bail!("Target is not a legacy P2PKH address")
    }
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

pub fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn parse_language(lang: &str) -> Result<Language> {
    match lang.to_lowercase().as_str() {
        "english"             => Ok(Language::English),
        "portuguese"          => Ok(Language::Portuguese),
        "spanish"             => Ok(Language::Spanish),
        "french"              => Ok(Language::French),
        "italian"             => Ok(Language::Italian),
        "czech"               => Ok(Language::Czech),
        "korean"              => Ok(Language::Korean),
        "japanese"            => Ok(Language::Japanese),
        "chinese-simplified"  => Ok(Language::SimplifiedChinese),
        "chinese-traditional" => Ok(Language::TraditionalChinese),
        _ => anyhow::bail!(
            "Unknown language '{lang}'. Supported: english, portuguese, spanish, \
             french, italian, czech, korean, japanese, chinese-simplified, \
             chinese-traditional"
        ),
    }
}
