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
    /// Optional only when --selftest is given.
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
    ///   Exactly one alternative is chosen per slot.
    /// • Within an alternative, WORDS are separated by commas.
    /// • A bare '?' inside an alternative marks a MISSING word whose value
    ///   is searched from the full BIP-39 wordlist — but ONLY within that slot.
    ///
    /// EXAMPLE (3 slots, total 3+2+2 = 7 known words + 1 missing in slot 1)
    ///   zebra,liquid,tornado,?   <- slot 1: 3 words + 1 unknown
    ///   orbit,galaxy             <- slot 2: 2 words
    ///   venture,sun              <- slot 3: 2 words
    ///
    /// The total known+unknown word count across all chosen slots must equal 12
    /// (BIP-39 mnemonic length).
    #[arg(long, value_name = "FILE")]
    tokenlist: Option<PathBuf>,

    /// Keep SLOT order as written in the file (do not permute slots).
    /// By default slots are permuted: 3 slots → 3! = 6 orderings tried.
    #[arg(long)]
    keep_token_order: bool,

    /// Keep WORD order within each slot as written (do not permute words inside a slot).
    /// By default words within a slot are permuted independently of other slots.
    /// A '?' marker keeps its position when this flag is set.
    #[arg(long)]
    keep_word_order: bool,

    /// Minimum number of slots to use from the tokenlist (default: all slots).
    /// When set below the total, combinations of that many slots are tried.
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
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A single token inside an alternative: either a concrete BIP-39 word or a
/// wildcard `?` whose value is searched from the full wordlist.
#[derive(Debug, Clone)]
enum Token {
    Word(String),
    Missing,
}

/// One alternative = an ordered list of tokens (words / wildcards).
type Alternative = Vec<Token>;

/// One slot = a set of mutually-exclusive alternatives.
/// Exactly one alternative is chosen per slot during the search.
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
            return Ok(());
        } else {
            anyhow::bail!("One or more selftests FAILED");
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
        .context("--tokenlist is required (classic word mode not yet wired to new slot logic)")?;

    let slots = parse_tokenlist(tokenlist_path)?;
    validate_slots(&slots, &args, language)?;

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

/// Parses the tokenlist file into `Vec<Slot>`.
///
/// Grammar recap:
///   line      ::= alternative (WS+ alternative)*
///   alternative ::= token (',' token)*
///   token     ::= '?' | <bip39_word>
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

/// Sanity-check: verify every alternative's words exist in the BIP-39 wordlist.
fn validate_slots(slots: &[Slot], _args: &Args, language: Language) -> Result<()> {
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
// Candidate generation — the heart of the new logic
// ---------------------------------------------------------------------------

/// Expands a single alternative into concrete (words, missing_count) pairs.
///
/// `missing_count` is the number of `?` tokens in the alternative.
/// The returned `Vec<String>` contains only the known words, in their original
/// positions relative to each other (gaps for `?` are tracked by index).
///
/// Returns: `(known_words, missing_count, positions_of_missing)`
///
/// `positions_of_missing` is a sorted list of indices into the *full* sequence
/// (known + missing interleaved) where missing words should be inserted.
/// This allows position-aware filling later.
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

/// Given a slot's chosen alternative (as known word indices + missing positions),
/// enumerate all concrete 12-slot-length word-index sequences for this slot.
///
/// `keep_word_order` — if true, only the missing word values are varied (positions fixed).
///                     if false, the known words are also permuted among themselves
///                     (the missing words stay in their declared positions relative
///                      to the permuted known words).
///
/// Returns an iterator of `Vec<u16>` each of length = `alt.len()`.
fn slot_candidates(
    known_indices: &[u16],
    missing_positions: &[usize],
    total_len: usize,       // known + missing
    keep_word_order: bool,
    wordlist_len: usize,
) -> Vec<Vec<u16>> {
    let missing_count = missing_positions.len();

    // Generate all permutations of known words (or just the one original order).
    let known_perms: Box<dyn Iterator<Item = Vec<u16>>> = if keep_word_order {
        Box::new(std::iter::once(known_indices.to_vec()))
    } else {
        let n = known_indices.len();
        Box::new(
            known_indices
                .iter()
                .copied()
                .permutations(n),
        )
    };

    let mut results: Vec<Vec<u16>> = Vec::new();

    for known_perm in known_perms {
        // For each permutation of known words, enumerate all values for missing slots.
        // Missing positions are fixed (positional meaning is preserved).
        for missing_values in (0..wordlist_len as u16)
            .permutations_with_replacement(missing_count)
        {
            // Reconstruct the full sequence of length `total_len`.
            let mut seq: Vec<u16> = Vec::with_capacity(total_len);
            let mut ki = 0usize; // index into known_perm
            let mut mi = 0usize; // index into missing_values
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
// Helper: Cartesian product over slots, respecting alternatives
// ---------------------------------------------------------------------------

/// For a chosen list of slots, produce every combination of:
///   - one alternative per slot
///   - (optionally) permuted slot order
///   - (optionally) permuted word order within each slot
///
/// Returns a fully-collected `Vec<Vec<u16>>` where each inner Vec is a complete
/// phrase expressed as BIP-39 word indices.
/// Using Vec (not an iterator) avoids Rust lifetime issues with nested closures
/// borrowing `slot_alt_candidates`.
fn enumerate_phrases(
    chosen_slots: &[&Slot],
    keep_token_order: bool,
    keep_word_order: bool,
    wordlist: &'static [&'static str],
) -> Vec<Vec<u16>> {
    let wordlist_len = wordlist.len();

    // Step 1: For each slot x alternative, compute all concrete word-index sequences.
    // Layout: slot_alt_candidates[slot_idx][alt_idx] = Vec<Vec<u16>>
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
                            wordlist.iter().position(|x| *x == w.as_str()).unwrap() as u16
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

    // Step 2: Enumerate slot orderings (all permutations, or fixed single order).
    let num_slots = chosen_slots.len();
    let slot_orderings: Vec<Vec<usize>> = if keep_token_order {
        vec![(0..num_slots).collect()]
    } else {
        (0..num_slots).permutations(num_slots).collect()
    };

    // Step 3: For each slot ordering x alt combination x per-slot candidate,
    // concatenate into a single flat phrase and push to results.
    let mut results: Vec<Vec<u16>> = Vec::new();

    for order in &slot_orderings {
        // Cartesian product: choose one alternative index per slot in this order.
        let alt_counts: Vec<usize> = order
            .iter()
            .map(|&si| slot_alt_candidates[si].len())
            .collect();

        for alt_indices in alt_counts.iter().map(|&n| 0..n).multi_cartesian_product() {
            // Collect the per-slot candidate lists for this alt-combination.
            let per_slot: Vec<&Vec<Vec<u16>>> = order
                .iter()
                .zip(alt_indices.iter())
                .map(|(&si, &ai)| &slot_alt_candidates[si][ai])
                .collect();

            // Cartesian product across per-slot candidate lists, then concatenate.
            let cand_counts: Vec<usize> = per_slot.iter().map(|c| c.len()).collect();
            for cand_indices in cand_counts.iter().map(|&n| 0..n).multi_cartesian_product() {
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
// Permutations-with-replacement helper (not in itertools by default)
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
            return PermWithReplacement { pool, indices: vec![], k, first: true, done: k != 0 };
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
            return Some(self.indices.iter().map(|&i| self.pool[i].clone()).collect());
        }
        // Increment: rightmost digit that hasn't reached pool.len()-1.
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
// GPU search
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
    let batch_size = 1 << 20;

    println!("Using GPU (CUDA) for tokenlist search.");

    let min_token = args.min_token.unwrap_or(slots.len()).min(slots.len());
    let max_token = slots.len();
    println!("Trying slot subsets {min_token}..={max_token}.");

    let mut total_checked: usize = 0;

    for slot_count in min_token..=max_token {
        let slot_indices: Vec<usize> = (0..slots.len()).collect();

        for chosen_indices in slot_indices.iter().copied().combinations(slot_count) {
            let chosen: Vec<&Slot> = chosen_indices.iter().map(|&i| &slots[i]).collect();

            // Kumpulkan SEMUA phrase untuk kombinasi slot ini sekaligus
            let all_phrases: Vec<[u16; 12]> = enumerate_phrases(
                &chosen,
                args.keep_token_order,
                args.keep_word_order,
                wordlist,
            )
            .into_iter()
            .filter(|p| p.len() == 12)
            .map(|p| {
                let mut a = [0u16; 12];
                a.copy_from_slice(&p);
                a
            })
            .collect();

            if all_phrases.is_empty() {
                continue;
            }

            let phrase_count = all_phrases.len();
            println!(
                "Slot combination {:?}: {} phrase(s) to check on GPU.",
                chosen_indices,
                format_number(phrase_count)
            );

            // Kirim seluruh batch sekaligus ke GPU
            if let Some(hit) =
                gpu_handle.search(all_phrases.into_iter(), &gpu_wordlist, &target_h160, batch_size)?
            {
                let phrase: Vec<&str> =
                    hit.indices.iter().map(|&i| wordlist[i as usize]).collect();
                println!("Found matching mnemonic: {}", phrase.join(" "));
                println!("Candidate index (0-based): {}", hit.global_index);
                println!("Derived address: {target}");
                return Ok(true);
            }

            total_checked += phrase_count;
            println!(
                "Checked ~{} candidates total so far...",
                format_number(total_checked)
            );
            let _ = io::stdout().flush();
        }
    }

    Ok(false)
}

// ---------------------------------------------------------------------------
// CPU search
// ---------------------------------------------------------------------------

fn run_search_cpu(
    args: &Args,
    slots: &[Slot],
    target: &Address<NetworkChecked>,
    language: Language,
) -> Result<bool> {
    let num_threads = if args.threads == 0 { num_cpus::get() } else { args.threads };
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

            let phrases: Vec<Vec<u16>> = enumerate_phrases(
                &chosen,
                args.keep_token_order,
                args.keep_word_order,
                wordlist,
            )
            .into_iter()
            .filter(|p: &Vec<u16>| p.len() == 12)
            .collect();

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

                let mnemonic = match Mnemonic::parse_in_normalized(language, &phrase_str) {
                    Ok(m) => m,
                    Err(_) => return,
                };

                let seed = mnemonic.to_seed("");
                let master_xprv = match Xpriv::new_master(Network::Bitcoin, &seed) {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let child_xprv = match master_xprv.derive_priv(&secp, &derivation_path) {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let child_pub = PublicKey::new(child_xprv.private_key.public_key(&secp));
                let addr: Address<NetworkChecked> = Address::p2pkh(&child_pub, Network::Bitcoin);

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
// P2PKH helpers
// ---------------------------------------------------------------------------

fn p2pkh_hash160(addr: &Address<NetworkChecked>) -> Result<[u8; 20]> {
    let spk = addr.script_pubkey();
    let bytes = spk.as_bytes();
    if bytes.len() == 25 && bytes[0] == 0x76 && bytes[1] == 0xa9 && bytes[2] == 0x14 {
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
        "english"              => Ok(Language::English),
        "portuguese"           => Ok(Language::Portuguese),
        "spanish"              => Ok(Language::Spanish),
        "french"               => Ok(Language::French),
        "italian"              => Ok(Language::Italian),
        "czech"                => Ok(Language::Czech),
        "korean"               => Ok(Language::Korean),
        "japanese"             => Ok(Language::Japanese),
        "chinese-simplified"   => Ok(Language::SimplifiedChinese),
        "chinese-traditional"  => Ok(Language::TraditionalChinese),
        _ => anyhow::bail!(
            "Unknown language '{lang}'. Supported: english, portuguese, spanish, french, \
             italian, czech, korean, japanese, chinese-simplified, chinese-traditional"
        ),
    }
}
