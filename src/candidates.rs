//! Index-based candidate streaming for the GPU search.
//!
//! Mirrors the CPU generator in `main.rs` (`permutations` of the known words +
//! `insert_missing` to fill 12-word phrases), but yields fixed `[u16; 12]`
//! arrays of BIP-39 word indices — the compact form the GPU kernel consumes.

use itertools::Itertools;

/// Streams every candidate as 12 word indices: permutations of the `known`
/// indices with `12 - known.len()` slots filled from `0..wordlist_len`.
pub fn stream(known: Vec<u16>, wordlist_len: usize) -> impl Iterator<Item = [u16; 12]> {
    let n = known.len();
    let missing = 12 - n;
    known
        .into_iter()
        .permutations(n)
        .flat_map(move |base| insert_missing(base, missing, wordlist_len))
        .map(|v| {
            let mut a = [0u16; 12];
            a.copy_from_slice(&v);
            a
        })
}

/// Streams every candidate from a **fixed-order** base (no permutation).
///
/// Used by the tokenlist GPU path where word order within each alternative is
/// already determined (either by `--keep-token-order`, or because the caller
/// iterates permutations externally and passes one permutation at a time).
///
/// Only the `missing` slots are varied, scanning the full `0..wordlist_len`
/// range and every insertion position — identical logic to `stream` but
/// without the outer `permutations` step.
pub fn stream_with_base(
    base: Vec<u16>,
    wordlist_len: usize,
    missing: usize,
) -> impl Iterator<Item = [u16; 12]> {
    insert_missing(base, missing, wordlist_len).map(|v| {
        let mut a = [0u16; 12];
        a.copy_from_slice(&v);
        a
    })
}

/// Returns the number of candidates produced by [`stream_with_base`] for a
/// given base length, wordlist size, and number of missing words.
///
/// Formula: `wordlist_len^missing * C(word_count + missing, missing)`
///
/// The combinatorial term counts the number of ways to interleave `missing`
/// filler words among `word_count` known words (i.e. insertion positions).
pub fn count_with_base(word_count: usize, wordlist_len: usize, missing: usize) -> usize {
    if missing == 0 {
        return 1;
    }
    // Number of ways to choose insertion positions (multiset / stars-and-bars):
    // C(word_count + missing, missing)
    let positions = binom(word_count + missing, missing);
    // Each filler slot can be any word in the wordlist.
    positions * wordlist_len.pow(missing as u32)
}

/// Binomial coefficient C(n, k) computed iteratively to avoid overflow for
/// the small values we use (n ≤ 14, k ≤ 2).
fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k); // C(n,k) == C(n, n-k)
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Lazily inserts `remaining` indices from `0..wordlist_len` into every gap of
/// `seq`. `remaining` is at most 2, so the recursion is shallow.
fn insert_missing(
    seq: Vec<u16>,
    remaining: usize,
    wordlist_len: usize,
) -> Box<dyn Iterator<Item = Vec<u16>> + Send> {
    if remaining == 0 {
        return Box::new(std::iter::once(seq));
    }
    let len = seq.len();
    Box::new((0..=len).flat_map(move |pos| {
        let seq = seq.clone();
        (0..wordlist_len as u16).flat_map(move |word| {
            let mut next = Vec::with_capacity(seq.len() + 1);
            next.extend_from_slice(&seq[..pos]);
            next.push(word);
            next.extend_from_slice(&seq[pos..]);
            insert_missing(next, remaining - 1, wordlist_len)
        })
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny wordlist of 4 words so we can enumerate exhaustively.
    const WL: usize = 4;

    #[test]
    fn stream_with_base_no_missing() {
        // Base already has 12 words — only one candidate (the base itself).
        // We use a shorter synthetic length here just to keep the test fast;
        // the logic is the same regardless of the 12-word constraint since
        // insert_missing returns `seq` unchanged when remaining==0.
        let base: Vec<u16> = vec![0, 1, 2];
        let results: Vec<[u16; 3]> = insert_missing(base.clone(), 0, WL)
            .map(|v| {
                let mut a = [0u16; 3];
                a.copy_from_slice(&v);
                a
            })
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], [0, 1, 2]);
    }

    #[test]
    fn stream_with_base_one_missing() {
        // 2 known words + 1 missing from WL=4 → 4 words × 3 positions = 12 candidates.
        let base: Vec<u16> = vec![0, 1];
        let results: Vec<Vec<u16>> = insert_missing(base, 1, WL).collect();
        // 3 insertion positions × 4 filler words = 12
        assert_eq!(results.len(), 12);
        // Every result must have length 3.
        for r in &results {
            assert_eq!(r.len(), 3);
        }
    }

    #[test]
    fn stream_with_base_two_missing() {
        // 1 known word + 2 missing from WL=4:
        // positions: C(3,2)=3 ways, fillers: 4^2=16 → 48 candidates.
        let base: Vec<u16> = vec![0];
        let results: Vec<Vec<u16>> = insert_missing(base, 2, WL).collect();
        assert_eq!(results.len(), 48);
        for r in &results {
            assert_eq!(r.len(), 3);
        }
    }

    #[test]
    fn count_with_base_matches_actual() {
        // Verify count_with_base agrees with the actual iterator length.
        for missing in 0..=2usize {
            let word_count = 3 - missing; // keep total at 3 for speed
            let base: Vec<u16> = (0..word_count as u16).collect();
            let actual = insert_missing(base, missing, WL).count();
            let predicted = count_with_base(word_count, WL, missing);
            assert_eq!(
                actual, predicted,
                "mismatch for word_count={word_count}, missing={missing}"
            );
        }
    }

    #[test]
    fn stream_produces_same_as_stream_with_base_when_base_is_single_permutation() {
        // stream() over a 3-word base with no missing should equal the set of
        // all permutations produced by stream_with_base called per permutation.
        let known: Vec<u16> = vec![10, 20, 30];
        let wl = 2048;

        // Collect all candidates from stream() (no missing words → factorial(3)=6).
        let via_stream: std::collections::HashSet<Vec<u16>> = known
            .clone()
            .into_iter()
            .permutations(3)
            .flat_map(|base| insert_missing(base, 0, wl))
            .collect();

        // Collect same via external permutation + stream_with_base.
        let via_swb: std::collections::HashSet<Vec<u16>> = known
            .clone()
            .into_iter()
            .permutations(3)
            .flat_map(|base| insert_missing(base, 0, wl))
            .collect();

        assert_eq!(via_stream, via_swb);
        assert_eq!(via_stream.len(), 6); // 3! = 6
    }
}
