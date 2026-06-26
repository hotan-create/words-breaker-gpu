//! Index-based candidate streaming for the GPU search.
//!
//! Yields fixed `[u16; 12]` arrays of BIP-39 word indices consumed by the GPU kernel.

use itertools::Itertools;

/// Streams every candidate as 12 word indices.
/// Permutes the `known` indices AND fills `12 - known.len()` slots from `0..wordlist_len`.
/// Used by the classic (non-tokenlist) GPU path (`gpu.rs`).
#[allow(dead_code)]
pub fn stream(known: Vec<u16>, wordlist_len: usize) -> impl Iterator<Item = [u16; 12]> {
    let n = known.len();
    let missing = 12 - n;
    known
        .into_iter()
        .permutations(n)
        .flat_map(move |base| fill_missing_at_end(base, missing, wordlist_len))
        .map(|v| {
            let mut a = [0u16; 12];
            a.copy_from_slice(&v);
            a
        })
}

/// Appends `remaining` filler indices (each from `0..wordlist_len`) to `seq`.
fn fill_missing_at_end(
    seq: Vec<u16>,
    remaining: usize,
    wordlist_len: usize,
) -> Box<dyn Iterator<Item = Vec<u16>> + Send> {
    if remaining == 0 {
        return Box::new(std::iter::once(seq));
    }
    Box::new((0..wordlist_len as u16).flat_map(move |word| {
        let mut next = seq.clone();
        next.push(word);
        fill_missing_at_end(next, remaining - 1, wordlist_len)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WL: usize = 4;

    #[test]
    fn no_missing_yields_one() {
        let base: Vec<u16> = vec![0, 1, 2];
        let results: Vec<Vec<u16>> = fill_missing_at_end(base, 0, WL).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], vec![0, 1, 2]);
    }

    #[test]
    fn one_missing_yields_wl() {
        let base: Vec<u16> = vec![0, 1];
        let results: Vec<Vec<u16>> = fill_missing_at_end(base, 1, WL).collect();
        assert_eq!(results.len(), WL);
        for r in &results {
            assert_eq!(r.len(), 3);
            assert_eq!(r[0], 0);
            assert_eq!(r[1], 1);
        }
    }

    #[test]
    fn two_missing_yields_wl_squared() {
        let base: Vec<u16> = vec![5];
        let results: Vec<Vec<u16>> = fill_missing_at_end(base, 2, WL).collect();
        assert_eq!(results.len(), WL * WL);
        for r in &results {
            assert_eq!(r.len(), 3);
            assert_eq!(r[0], 5);
        }
    }
}
