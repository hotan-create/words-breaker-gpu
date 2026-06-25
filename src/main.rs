use anyhow::{Context, Result};
use bip39::{Language, Mnemonic};
use bitcoin::address::{Address, NetworkChecked, NetworkUnchecked};
use bitcoin::bip32::{DerivationPath, Xpriv};
use bitcoin::secp256k1::Secp256k1;
use bitcoin::{Network, PublicKey};
use clap::Parser;
use itertools::Itertools;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

mod candidates;
mod gpu;

#[derive(Parser, Debug)]
#[command(
    about = "Try permutations of BIP-39 words (10-12) to match a BTC legacy address. \
             Words can be supplied directly or via --tokenlist. \
             Missing words (when fewer than 12 are given) are filled from the 2048-word BIP-39 list.",
    version
)]
struct Args {
    /// Target legacy Bitcoin address (Base58, starting with '1').
    /// Optional only when --selftest is given.
    target_address: Option<String>,

    /// 10, 11, or 12 words (unordered or partially ordered). Ignored when
    /// --tokenlist is used.
    words: Vec<String>,

    /// Path to a tokenlist file.
    ///
    /// Format (one token-group per line):
    ///   word1,word2,word3  alt1,alt2,alt3
    ///
    /// Each line defines one "slot" in the phrase. A slot contains one or more
    /// SPACE-separated alternatives; exactly one alternative is chosen per slot
    /// (mutual exclusion). An alternative is a COMMA-separated list of words
    /// that are treated as a single ordered group.
    ///
    /// Example line:
    ///   orbit,galaxy,venture,sun  orbit,galaxy,sun,venture
    ///
    /// This slot contributes either ["orbit","galaxy","venture","sun"] or
    /// ["orbit","galaxy","sun","venture"] to the phrase — never both.
    ///
    /// The number of words across all chosen alternatives must total 12, or
    /// equal 10/11 so that missing words are filled from the BIP-39 wordlist.
    #[arg(long, value_name = "FILE")]
    tokenlist: Option<PathBuf>,

    /// When set, the words within each chosen alternative keep their original
    /// order (no permutations). Without this flag every alternative is fully
    /// permuted before being tested.
    #[arg(long)]
    keep_token_order: bool,

    /// Minimum number of tokens (slots) to use from the tokenlist.
    /// Slots beyond this minimum are optional and tried in combination.
    /// Defaults to the total number of slots in the file.
    #[arg(long, value_name = "N")]
    min_token: Option<usize>,

    /// BIP-39 wordlist language
    #[arg(long, short, default_value = "english")]
    language: String,

    /// Number of threads to use (defaults to number of CPU cores)
    #[arg(long, short, default_value_t = 0)]
    threads: usize,

    /// Verify each GPU crypto primitive against the CPU reference and exit.
    #[arg(long)]
    selftest: bool,

    /// Force CPU (rayon) search instead of GPU.
    #[arg(long)]
    cpu: bool,
}

// ---------------------------------------------------------------------------
// Token-list types
// ---------------------------------------------------------------------------

/// One alternative within a slot: an ordered list of BIP-39 words.
type Alternative = Vec<String>;

/// One slot: a set of mutually-exclusive alternatives.
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

    let target_address_unchecked = target_address
        .parse::<Address<NetworkUnchecked>>()
        .context("Invalid target Bitcoin address")?;

    let target_address: Address<NetworkChecked> = target_address_unchecked
        .require_network(Network::Bitcoin.into())
        .context("This tool currently only supports mainnet legacy addresses")?;

    let language = parse_language(&args.language)?;
    let start = Instant::now();

    let found = if let Some(ref tokenlist_path) = args.tokenlist {
        // ── tokenlist mode ──────────────────────────────────────────────────
        let slots = parse_tokenlist(tokenlist_path)?;
        run_tokenlist_search(&args, &slots, &target_address, language)?
    } else {
        // ── classic words mode ───────────────────────────────────────────────
        if !(10..=12).contains(&args.words.len()) {
            anyhow::bail!("Expected 10, 11, or 12 words, got {}", args.words.len());
        }
        if args.cpu {
            run_cpu_search(&args, &target_address, language)?
        } else {
            match search_permutations_gpu(&args.words, &target_address, language) {
                Ok(found) => found,
                Err(e) => {
                    eprintln!("GPU search unavailable ({e:#}); falling back to CPU.");
                    run_cpu_search(&args, &target_address, language)?
                }
            }
        }
    };

    let elapsed = start.elapsed();
    if !found {
        println!(
            "Exhausted all candidates without a match (elapsed: {:?})",
            elapsed
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tokenlist parsing
// ---------------------------------------------------------------------------

/// Parses `tokenlist.txt` into a `Vec<Slot>`.
///
/// File format:
///   • One line = one slot.
///   • Blank lines and lines starting with `#` are ignored.
///   • Within a line, alternatives are separated by one or more spaces/tabs.
///   • Within an alternative, words are separated by commas.
///
/// Example line:
///   orbit,galaxy,venture,sun orbit,galaxy,sun,venture
///
/// Yields one slot with two alternatives:
///   [["orbit","galaxy","venture","sun"], ["orbit","galaxy","sun","venture"]]
fn parse_tokenlist(path: &PathBuf) -> Result<Vec<Slot>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read tokenlist file: {}", path.display()))?;

    let mut slots: Vec<Slot> = Vec::new();

    for (lineno, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Split alternatives by whitespace
        let alternatives: Vec<Alternative> = line
            .split_whitespace()
            .map(|alt| {
                alt.split(',')
                    .filter(|w| !w.is_empty())
                    .map(|w| w.trim().to_string())
                    .collect::<Alternative>()
            })
            .filter(|alt| !alt.is_empty())
            .collect();

        if alternatives.is_empty() {
            eprintln!("Warning: line {} is empty after parsing, skipping.", lineno + 1);
            continue;
        }

        slots.push(alternatives);
    }

    if slots.is_empty() {
        anyhow::bail!("Tokenlist file is empty or contains no valid lines");
    }

    println!("Loaded {} slot(s) from tokenlist.", slots.len());
    Ok(slots)
}

// ---------------------------------------------------------------------------
// Tokenlist search orchestration
// ---------------------------------------------------------------------------

/// Entry point for tokenlist-mode search.
///
/// Strategy:
/// 1. Determine which subset of slots to use (min_token..=total_slots).
/// 2. For each subset combination, pick one alternative per slot (Cartesian
///    product across slots).
/// 3. Flatten the chosen alternatives into a word list.
/// 4. Unless --keep-token-order, permute the word list.
/// 5. Try the resulting phrase against the target address.
fn run_tokenlist_search(
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
    println!("Using CPU with {} threads (tokenlist mode)", num_threads);

    let min_token = args.min_token.unwrap_or(slots.len()).min(slots.len());
    let max_token = slots.len();

    // Validate that at least one slot-count can produce valid phrase lengths.
    // (We do this lazily per phrase below, but a quick heads-up is useful.)
    println!(
        "Trying slot subsets of size {}..={} (--min-token={}).",
        min_token, max_token, min_token
    );

    let wordlist: &'static [&'static str] = language.words_by_prefix("");
    let target_str = target.to_string();
    let derivation_path: DerivationPath = "m/44'/0'/0'/0/0".parse()?;
    let secp = Arc::new(Secp256k1::new());

    let counter = Arc::new(AtomicUsize::new(0));
    let found = Arc::new(AtomicBool::new(false));
    let found_phrase = Arc::new(std::sync::Mutex::new(String::new()));
    let found_index = Arc::new(AtomicUsize::new(0));

    // Iterate over every valid subset size (min_token..=max_token).
    'outer: for slot_count in min_token..=max_token {
        if found.load(Ordering::Relaxed) {
            break;
        }

        // Generate every combination of `slot_count` slots from the full list.
        let slot_indices: Vec<usize> = (0..slots.len()).collect();

        for chosen_slot_indices in slot_indices.iter().copied().combinations(slot_count) {
            if found.load(Ordering::Relaxed) {
                break 'outer;
            }

            let chosen_slots: Vec<&Slot> =
                chosen_slot_indices.iter().map(|&i| &slots[i]).collect();

            // Build the stream of word-lists via Cartesian product over alternatives.
            // Each element is a flat Vec<String> of all words from the chosen alts.
            let phrase_words_iter = cartesian_product_slots(&chosen_slots);

            phrase_words_iter
                .par_bridge()
                .for_each(|words: Vec<String>| {
                    if found.load(Ordering::Relaxed) {
                        return;
                    }

                    // Determine how many BIP-39 words are missing.
                    let word_count = words.len();
                    if word_count > 12 {
                        // Too many words — skip.
                        return;
                    }
                    let missing = 12 - word_count;
                    if word_count < 10 {
                        // Not enough words even with missing-fill — skip.
                        return;
                    }

                    // Produce candidate phrases (with or without permutation).
                    let candidates: Box<dyn Iterator<Item = String> + Send> =
                        if args.keep_token_order {
                            // Keep order: only vary the missing-word insertion positions.
                            Box::new(
                                insert_missing(words, missing, wordlist)
                                    .map(|v| v.join(" ")),
                            )
                        } else {
                            // Full permutation of all words + missing-word insertion.
                            let n = words.len();
                            Box::new(
                                words
                                    .into_iter()
                                    .permutations(n)
                                    .flat_map(move |base| {
                                        insert_missing(base, missing, wordlist)
                                            .map(|v| v.join(" "))
                                    }),
                            )
                        };

                    for phrase in candidates {
                        if found.load(Ordering::Relaxed) {
                            return;
                        }

                        let i = counter.fetch_add(1, Ordering::Relaxed);
                        if i % 100_000 == 0 && i > 0 {
                            println!("Checked {} candidates...", format_number(i));
                            let _ = io::stdout().flush();
                        }

                        let mnemonic = match Mnemonic::parse_in_normalized(language, &phrase) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };

                        let seed = mnemonic.to_seed("");
                        let master_xprv = match Xpriv::new_master(Network::Bitcoin, &seed) {
                            Ok(x) => x,
                            Err(_) => continue,
                        };
                        let child_xprv =
                            match master_xprv.derive_priv(&secp, &derivation_path) {
                                Ok(x) => x,
                                Err(_) => continue,
                            };
                        let child_priv = child_xprv.private_key;
                        let child_pub = PublicKey::new(child_priv.public_key(&secp));
                        let addr: Address<NetworkChecked> =
                            Address::p2pkh(&child_pub, Network::Bitcoin);

                        if addr.to_string() == target_str {
                            found.store(true, Ordering::SeqCst);
                            found_index.store(i, Ordering::SeqCst);
                            let mut fp = found_phrase.lock().unwrap();
                            *fp = phrase;
                            return;
                        }
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
        println!("Found matching mnemonic: {}", *fp);
        println!("Candidate index (0-based): {}", idx);
        println!("Derived address: {}", target_str);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Generates the Cartesian product over a list of slots (each slot is a list
/// of alternatives). Yields flat `Vec<String>` — one per combination — where
/// each alternative's words are appended in order.
///
/// This is a lazy iterator: no allocation beyond the current combination.
fn cartesian_product_slots<'a>(
    slots: &'a [&'a Slot],
) -> impl Iterator<Item = Vec<String>> + 'a {
    // Build a Vec of Vec<&Alternative> then multi_cartesian_product.
    let alt_lists: Vec<&[Alternative]> = slots.iter().map(|s| s.as_slice()).collect();

    alt_lists
        .into_iter()
        .multi_cartesian_product()
        .map(|chosen_alts: Vec<&Alternative>| {
            chosen_alts.iter().flat_map(|alt| alt.iter().cloned()).collect()
        })
}

// ---------------------------------------------------------------------------
// Helpers reused from original code (unchanged)
// ---------------------------------------------------------------------------

fn run_cpu_search(
    args: &Args,
    target_address: &Address<NetworkChecked>,
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
    println!("Using CPU with {} threads", num_threads);
    search_permutations_parallel(&args.words, target_address, language)
}

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

fn search_permutations_gpu(
    words: &[String],
    target: &Address<NetworkChecked>,
    language: Language,
) -> Result<bool> {
    let gpu = gpu::Gpu::new()?;
    println!("Using GPU (CUDA)");

    let wordlist: &'static [&'static str] = language.words_by_prefix("");
    let gpu_wordlist = gpu::GpuWordlist::new(wordlist)?;
    let target_h160 = p2pkh_hash160(target)?;

    let mut known_idx: Vec<u16> = Vec::with_capacity(words.len());
    for w in words {
        let pos = wordlist
            .iter()
            .position(|x| *x == w)
            .with_context(|| format!("Word '{w}' is not in the BIP-39 wordlist"))?;
        known_idx.push(pos as u16);
    }

    let missing = 12 - words.len();
    if missing > 0 {
        println!(
            "Got {} words; completing {} missing word(s) from the {}-word BIP-39 list.",
            words.len(),
            missing,
            wordlist.len()
        );
    }
    let total = total_candidates(words.len(), wordlist.len(), missing);
    println!(
        "Searching {} candidates on GPU (streamed)...",
        format_number(total)
    );

    let candidates = candidates::stream(known_idx, wordlist.len());
    let batch_size = 1 << 20;
    let hit = gpu.search(candidates, &gpu_wordlist, &target_h160, batch_size)?;

    match hit {
        Some(h) => {
            let phrase: Vec<&str> = h.indices.iter().map(|&i| wordlist[i as usize]).collect();
            let phrase = phrase.join(" ");
            println!("Found matching mnemonic: {}", phrase);
            if missing > 0 {
                let recovered = recovered_words(words, &phrase);
                println!("Recovered missing word(s): {}", recovered.join(" "));
            }
            println!("Candidate index (0-based): {}", h.global_index);
            println!("Derived address: {}", target);
            Ok(true)
        }
        None => Ok(false),
    }
}

pub fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn parse_language(lang: &str) -> Result<Language> {
    match lang.to_lowercase().as_str() {
        "english" => Ok(Language::English),
        "portuguese" => Ok(Language::Portuguese),
        "spanish" => Ok(Language::Spanish),
        "french" => Ok(Language::French),
        "italian" => Ok(Language::Italian),
        "czech" => Ok(Language::Czech),
        "korean" => Ok(Language::Korean),
        "japanese" => Ok(Language::Japanese),
        "chinese-simplified" => Ok(Language::SimplifiedChinese),
        "chinese-traditional" => Ok(Language::TraditionalChinese),
        _ => anyhow::bail!(
            "Unknown language: {}. Supported: english, portuguese, spanish, french, \
             italian, czech, korean, japanese, chinese-simplified, chinese-traditional",
            lang
        ),
    }
}

fn search_permutations_parallel(
    words: &[String],
    target: &Address<NetworkChecked>,
    language: Language,
) -> Result<bool> {
    let derivation_path: DerivationPath = "m/44'/0'/0'/0/0".parse()?;
    let target_str = target.to_string();
    let secp = Arc::new(Secp256k1::new());
    let missing = 12 - words.len();
    let wordlist: &'static [&'static str] = language.words_by_prefix("");

    if missing > 0 {
        println!(
            "Got {} words; completing {} missing word(s) from the {}-word BIP-39 list.",
            words.len(),
            missing,
            wordlist.len()
        );
    }

    let total = total_candidates(words.len(), wordlist.len(), missing);
    println!(
        "Searching {} candidates (streamed, not held in memory)...",
        format_number(total)
    );
    let _ = io::stdout().flush();

    let owned_words: Vec<String> = words.to_vec();
    let candidates = owned_words
        .into_iter()
        .permutations(words.len())
        .flat_map(move |base| insert_missing(base, missing, wordlist).map(|v| v.join(" ")));

    let counter = Arc::new(AtomicUsize::new(0));
    let found = Arc::new(AtomicBool::new(false));
    let found_phrase = Arc::new(std::sync::Mutex::new(String::new()));
    let found_index = Arc::new(AtomicUsize::new(0));

    candidates.par_bridge().for_each(|phrase| {
        if found.load(Ordering::Relaxed) {
            return;
        }

        let i = counter.fetch_add(1, Ordering::Relaxed);
        if i % 100000 == 0 && i > 0 {
            println!("Checked {} candidates...", format_number(i));
            let _ = io::stdout().flush();
        }

        let mnemonic = match Mnemonic::parse_in_normalized(language, &phrase) {
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
        let child_priv = child_xprv.private_key;
        let child_pub = PublicKey::new(child_priv.public_key(&secp));
        let addr: Address<NetworkChecked> = Address::p2pkh(&child_pub, Network::Bitcoin);

        if addr.to_string() == target_str {
            found.store(true, Ordering::SeqCst);
            found_index.store(i, Ordering::SeqCst);
            let mut fp = found_phrase.lock().unwrap();
            *fp = phrase;
        }
    });

    if found.load(Ordering::SeqCst) {
        let fp = found_phrase.lock().unwrap();
        let idx = found_index.load(Ordering::SeqCst);
        println!("Found matching mnemonic: {}", *fp);
        if missing > 0 {
            let recovered = recovered_words(words, &fp);
            println!("Recovered missing word(s): {}", recovered.join(" "));
        }
        println!("Candidate index (0-based): {}", idx);
        println!("Derived address: {}", target_str);
        Ok(true)
    } else {
        Ok(false)
    }
}

fn total_candidates(n: usize, wordlist_len: usize, missing: usize) -> usize {
    let factorial: usize = (1..=n).product::<usize>().max(1);
    factorial * wordlist_len.pow(missing as u32)
}

fn insert_missing(
    seq: Vec<String>,
    remaining: usize,
    wordlist: &'static [&'static str],
) -> Box<dyn Iterator<Item = Vec<String>> + Send> {
    if remaining == 0 {
        return Box::new(std::iter::once(seq));
    }

    let len = seq.len();
    Box::new((0..=len).flat_map(move |pos| {
        let seq = seq.clone();
        wordlist.iter().flat_map(move |&word| {
            let mut next = Vec::with_capacity(seq.len() + 1);
            next.extend_from_slice(&seq[..pos]);
            next.push(word.to_string());
            next.extend_from_slice(&seq[pos..]);
            insert_missing(next, remaining - 1, wordlist)
        })
    }))
}

fn recovered_words(known: &[String], phrase: &str) -> Vec<String> {
    let mut remaining: Vec<String> = known.to_vec();
    let mut recovered = Vec::new();
    for word in phrase.split_whitespace() {
        if let Some(pos) = remaining.iter().position(|k| k == word) {
            remaining.remove(pos);
        } else {
            recovered.push(word.to_string());
        }
    }
    recovered
}
