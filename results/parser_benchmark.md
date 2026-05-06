# AV Parser Benchmark Results

**Output provenance:** llm_cached=10

**Overall accuracy: 10/10 = 100%**

| # | Diff. | Depth | Expected | Parser output | Source | Pass |
|---|-------|-------|----------|---------------|--------|------|
| 1 | easy | 1 | `L & ~P` | `L & ~P` | llm_cached | &#10003; |
| 2 | easy | 1 | `~L` | `~L` | llm_cached | &#10003; |
| 3 | easy | 1 | `P` | `P` | llm_cached | &#10003; |
| 4 | easy | 1 | `E` | `E` | llm_cached | &#10003; |
| 5 | easy | 1 | `~S` | `~S` | llm_cached | &#10003; |
| 6 | medium | 2 | `L & ~P & G & R` | `L & ~P & G & R` | llm_cached | &#10003; |
| 7 | medium | 2 | `E | ~L` | `E | ~L` | llm_cached | &#10003; |
| 8 | medium | 3 | `L & ~P & G & R & S` | `L & ~P & G & R & S` | llm_cached | &#10003; |
| 9 | hard | 3 | `(L & ~P) & ~E` | `(L & ~P) & ~E` | llm_cached | &#10003; |
| 10 | hard | 4 | `(L & ~P & G & R & S) & ~E` | `(L & ~P & G & R & S) & ~E` | llm_cached | &#10003; |
