# Words Breaker

A command-line tool that attempts to recover a BIP-39 mnemonic seed phrase by testing permutations of 12 known words against a target Bitcoin legacy address.

## Use Case

If you have 12 BIP-39 mnemonic words but don't remember the correct order, this tool will brute-force permutations to find the combination that derives to your known Bitcoin address.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (1.70 or later recommended)
- An **NVIDIA GPU** plus the **CUDA toolkit** (`nvcc`) — the search runs on the GPU
  by default. The build compiles the CUDA kernels (`src/cuda/kernels.cu`) to PTX
  with `nvcc`. If no CUDA device is present at runtime, the tool falls back to the
  CPU automatically (or pass `--cpu`).

### CUDA library path

`cust` locates the CUDA driver library via `CUDA_LIBRARY_PATH`, which must point at
a CUDA root containing `lib64/` and `include/cuda.h`. On a distro-packaged CUDA
install (`nvcc` in `/usr/bin`) that is `/usr/lib/cuda`, and `.cargo/config.toml`
already sets it. For a standard toolkit install, export your root, e.g.:

```bash
export CUDA_LIBRARY_PATH=/usr/local/cuda
```

## Building

### Windows

```powershell
cargo build --release
```

The binary will be located at `target\release\words-breaker.exe`.

### Linux / macOS

```bash
cargo build --release
```

The binary will be located at `target/release/words-breaker`.

## Usage

```
words-breaker <TARGET_ADDRESS> <WORD1> <WORD2> ... <WORD12> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `TARGET_ADDRESS` | Target legacy Bitcoin (P2PKH) address (Base58, starting with `1`) |
| `WORD1..WORDN` | 10, 11, or 12 BIP-39 words in any order. With 10 or 11 words, the missing word(s) are completed from the 2048-word BIP-39 list. |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-l, --language` | `english` | BIP-39 wordlist language |
| `-t, --threads` | `0` (all cores) | CPU threads (CPU path only) |
| `--cpu` | off | Force the CPU (rayon) search instead of the GPU |
| `--selftest` | | Verify each GPU crypto primitive against the CPU reference and exit |
| `-h, --help` | | Print help |
| `-V, --version` | | Print version |

The search runs on the **GPU by default**, streaming candidates in batches. Each
batch is filtered by BIP-39 checksum on the GPU (a cheap pass that keeps ~1/16 of
candidates), then only the survivors run the full seed/derivation/address
pipeline. Use `--selftest` to confirm the GPU primitives (SHA-256/512, RIPEMD-160,
HMAC, PBKDF2, secp256k1, BIP32) match the reference CPU crates bit-for-bit.

**Supported languages:** `english`, `portuguese`, `spanish`, `french`, `italian`, `czech`, `korean`, `japanese`, `chinese-simplified`, `chinese-traditional`

### Examples

**Windows:**
```powershell
.\target\release\words-breaker.exe 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa abandon ability able about above absent absorb abstract absurd abuse access accident
```

**Linux / macOS:**
```bash
./target/release/words-breaker 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa abandon ability able about above absent absorb abstract absurd abuse access accident
```

**Verify the GPU implementation:**
```bash
./target/release/words-breaker --selftest
```

**With Portuguese wordlist:**
```bash
./target/release/words-breaker 1CfntEjWHwCc7moXnMHUX8QuBJaakAnv8U bexiga bonde curativo nevoeiro mundial vareta urubu megafone cozinha livro surpresa senador -l portuguese
```

## How It Works

1. Streams permutations of the provided words (completing missing words when 10/11
   are given), as compact 12-index arrays — nothing is held fully in memory
2. On the GPU, each candidate is checksum-filtered, then survivors are derived:
   PBKDF2-HMAC-SHA512 seed → BIP32 `m/44'/0'/0'/0/0` → secp256k1 public key →
   `RIPEMD160(SHA256(pubkey))`
3. The resulting hash160 is compared against the target address's hash160
4. Stops and outputs the correct phrase when a match is found

## Performance Notes

- 12 words have 479,001,600 (12!) possible permutations; supplying 10 or 11 words
  multiplies this by up to 2048 per missing word
- The whole space is streamed and searched (there is no fixed permutation cap)
- The dominant cost per candidate is PBKDF2-HMAC-SHA512 (2048 iterations), which is
  fixed by the BIP-39 spec; the GPU runs many candidates in parallel
- Invalid BIP-39 checksums are filtered out cheaply before the expensive work

## License

MIT
