# c1 pilot summary

Model `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` @ `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`, fp16, mps.
Generated 30 traces (gsm8k 15, math500 15), temperature 0.6 / top-p 0.95, max 4096 new tokens, one sample per problem.

Generation-prompt suffix: `er within \boxed{}.<｜Assistant｜><think>
`
`<think>` token ids [151648], `</think>` token ids [151649].
Hard check: 649 hits checked, 0 failures.

## gsm8k

### 1. Traces

- n = 15
- truncated (`length`): 3 (20.0%)
- reasoning incomplete (no `</think>`): 3 (20.0%)
- looped: 2 (13.3%)
- accuracy: 11/15 (73.3%)

### 2. Density

- hits per trace: mean 18.20, median 7.00, min 2.00, max 91.00
- hits per 1k reasoning tokens (per trace): mean 8.44, median 7.51, min 4.12, max 22.22
- pooled: 273 hits over 24924 reasoning tokens = 10.95 per 1k

### 3. Per-marker counts

| value | count | share |
|---|---|---|
| `wait` | 186 | 68.1% |
| `hmm` | 42 | 15.4% |
| `let me think` | 18 | 6.6% |
| `hold on` | 12 | 4.4% |
| `actually` | 9 | 3.3% |
| `no,` | 6 | 2.2% |

**Share of hits that are `wait`: 68.1%** (186/273)

### 4. Position

- paragraph-initial: 136/273 (49.8%)
- mid-token onsets: 137/273 (50.2%)

`prev_token_kind`:

| value | count | share |
|---|---|---|
| `newline` | 136 | 49.8% |
| `word` | 72 | 26.4% |
| `comma` | 30 | 11.0% |
| `period` | 27 | 9.9% |
| `other_punct` | 8 | 2.9% |

### 5. Gaps

Hits with a previous hit in the same trace: 258

| threshold | share of gaps >= | share of first-hit distances >= |
|---|---|---|
| 5 | 90.3% | 100.0% |
| 10 | 81.4% | 100.0% |
| 20 | 69.4% | 93.3% |
| 40 | 54.3% | 73.3% |

First-hit distance from reasoning start: mean 57.40, median 48.00, min 12.00, max 123.00

### 6. Clustering

- hits with another hit earlier in the same paragraph: 76/258 (29.5%)

### 7. Negatives pool

- positions with no hit in the next 100 tokens: **9043** (excluded for insufficient lookahead: 1500)
- positions with no hit in the next 60 tokens: **13535** (excluded for insufficient lookahead: 900)

By token kind at that position:

| kind | no hit in next 100 | no hit in next 60 |
|---|---|---|
| `newline` | 224 | 321 |
| `period` | 316 | 446 |
| `comma` | 554 | 837 |
| `other_punct` | 536 | 776 |
| `digit` | 1521 | 2302 |
| `word` | 5317 | 7978 |
| `other` | 575 | 875 |

### 8. Gate input — generated token ids that begin a marker

14 distinct token ids.

| token id | count |
|---|---|
| 14190 | 118 |
| 8580 | 33 |
| 3783 | 29 |
| 88190 | 27 |
| 80022 | 15 |
| 1077 | 13 |
| 3331 | 12 |
| 3520 | 8 |
| 902 | 6 |
| 13824 | 5 |
| 10061 | 3 |
| 6771 | 2 |
| 46936 | 1 |
| 33763 | 1 |

### 9. Paragraph openers the lexicon does not match

Listed only — nothing is added to the lexicon.

| rank | word | n | bigram | n |
|---|---|---|---|---|
| 1 | `so` | 105 | `so, the` | 25 |
| 2 | `but` | 71 | `final answer` | 12 |
| 3 | `alternatively` | 24 | `but wait` | 11 |
| 4 | `first` | 21 | `alternatively, if` | 10 |
| 5 | `then` | 13 | `i think` | 8 |
| 6 | `therefore` | 13 | `but he` | 7 |
| 7 | `final` | 12 | `but the` | 7 |
| 8 | `total` | 12 | `so, if` | 6 |
| 9 | `the` | 8 | `first, the` | 6 |
| 10 | `1` | 7 | `so, in` | 6 |
| 11 | `okay` | 7 | `so, yeah` | 6 |
| 12 | `alright` | 7 | `but, in` | 6 |
| 13 | `third` | 7 | `alright, so` | 5 |
| 14 | `if` | 5 | `but hold` | 5 |
| 15 | `12` | 5 | `just to` | 5 |
| 16 | `just` | 5 | `okay, so` | 5 |
| 17 | `second` | 5 | `now, the` | 5 |
| 18 | `fourth` | 5 | `third train` | 5 |
| 19 | `5` | 4 | `fourth train` | 5 |
| 20 | `yes` | 4 | `alternatively, maybe` | 4 |
| 21 | `next` | 4 | `so, total` | 4 |
| 22 | `john` | 3 | `therefore, the` | 4 |
| 23 | `he` | 3 | `then, the` | 4 |
| 24 | `week` | 3 | `yes, that` | 4 |
| 25 | `calculating` | 3 | `i don't` | 4 |
| 26 | `we` | 3 | `total cost` | 4 |
| 27 | `another` | 3 | `second train` | 4 |
| 28 | `divide` | 3 | `alternatively, perhaps` | 4 |
| 29 | `yep` | 3 | `but, given` | 4 |
| 30 | `1/2)c` | 3 | `the problem` | 3 |

### 10. Speed

- tokens/second: mean 63.92, median 64.23, min 61.80, max 64.94

## math500

### 1. Traces

- n = 15
- truncated (`length`): 5 (33.3%)
- reasoning incomplete (no `</think>`): 3 (20.0%)
- looped: 3 (20.0%)
- accuracy: 10/15 (66.7%)

### 2. Density

- hits per trace: mean 25.07, median 19.00, min 3.00, max 63.00
- hits per 1k reasoning tokens (per trace): mean 8.52, median 6.47, min 2.75, max 15.87
- pooled: 376 hits over 39393 reasoning tokens = 9.54 per 1k

### 3. Per-marker counts

| value | count | share |
|---|---|---|
| `wait` | 235 | 62.5% |
| `hmm` | 37 | 9.8% |
| `let me think` | 34 | 9.0% |
| `no,` | 25 | 6.6% |
| `hold on` | 24 | 6.4% |
| `actually` | 20 | 5.3% |
| `however` | 1 | 0.3% |

**Share of hits that are `wait`: 62.5%** (235/376)

### 4. Position

- paragraph-initial: 171/376 (45.5%)
- mid-token onsets: 205/376 (54.5%)

`prev_token_kind`:

| value | count | share |
|---|---|---|
| `newline` | 171 | 45.5% |
| `comma` | 66 | 17.6% |
| `period` | 66 | 17.6% |
| `word` | 60 | 16.0% |
| `other_punct` | 13 | 3.5% |

### 5. Gaps

Hits with a previous hit in the same trace: 361

| threshold | share of gaps >= | share of first-hit distances >= |
|---|---|---|
| 5 | 79.2% | 100.0% |
| 10 | 74.0% | 100.0% |
| 20 | 67.3% | 100.0% |
| 40 | 59.3% | 80.0% |

First-hit distance from reasoning start: mean 94.27, median 65.00, min 24.00, max 401.00

### 6. Clustering

- hits with another hit earlier in the same paragraph: 116/361 (32.1%)

### 7. Negatives pool

- positions with no hit in the next 100 tokens: **16824** (excluded for insufficient lookahead: 1500)
- positions with no hit in the next 60 tokens: **23618** (excluded for insufficient lookahead: 900)

By token kind at that position:

| kind | no hit in next 100 | no hit in next 60 |
|---|---|---|
| `newline` | 471 | 612 |
| `period` | 504 | 720 |
| `comma` | 1039 | 1534 |
| `other_punct` | 2103 | 2812 |
| `digit` | 3480 | 4514 |
| `word` | 8126 | 11897 |
| `other` | 1101 | 1529 |

### 8. Gate input — generated token ids that begin a marker

14 distinct token ids.

| token id | count |
|---|---|
| 14190 | 157 |
| 13824 | 43 |
| 3783 | 35 |
| 88190 | 26 |
| 902 | 25 |
| 1077 | 24 |
| 3331 | 23 |
| 3520 | 19 |
| 80022 | 11 |
| 6771 | 9 |
| 52592 | 1 |
| 4354 | 1 |
| 10061 | 1 |
| 47427 | 1 |

### 9. Paragraph openers the lexicon does not match

Listed only — nothing is added to the lexicon.

| rank | word | n | bigram | n |
|---|---|---|---|---|
| 1 | `so` | 109 | `but wait` | 19 |
| 2 | `but` | 97 | `therefore, the` | 15 |
| 3 | `therefore` | 42 | `but in` | 13 |
| 4 | `alternatively` | 36 | `so, in` | 13 |
| 5 | `first` | 20 | `so, the` | 12 |
| 6 | `then` | 15 | `final answer` | 12 |
| 7 | `final` | 12 | `but let` | 11 |
| 8 | `yes` | 10 | `alternatively, maybe` | 10 |
| 9 | `in` | 9 | `y` | 9 |
| 10 | `fill((0` | 9 | `alternatively, if` | 8 |
| 11 | `since` | 8 | `so, if` | 7 |
| 12 | `tan` | 8 | `a(1` | 7 |
| 13 | `which` | 7 | `tan x` | 6 |
| 14 | `1` | 7 | `alternatively, perhaps` | 6 |
| 15 | `factor` | 6 | `so, tan` | 6 |
| 16 | `another` | 6 | `alternatively, i` | 6 |
| 17 | `calculating` | 6 | `but the` | 6 |
| 18 | `given` | 6 | `given that` | 6 |
| 19 | `alright` | 5 | `which is` | 5 |
| 20 | `just` | 5 | `so, x` | 5 |
| 21 | `we` | 4 | `so, let's` | 5 |
| 22 | `thus` | 4 | `first, let` | 5 |
| 23 | `similarly` | 4 | `alright, so` | 5 |
| 24 | `the` | 4 | `now, the` | 5 |
| 25 | `for` | 4 | `fill((0, 4)--(3` | 5 |
| 26 | `to` | 3 | `just to` | 5 |
| 27 | `divide` | 3 | `2a^2` | 5 |
| 28 | `is` | 3 | `so, for` | 4 |
| 29 | `looking` | 3 | `factor tan` | 4 |
| 30 | `compute` | 3 | `a b` | 4 |

### 10. Speed

- tokens/second: mean 62.91, median 63.04, min 61.40, max 64.72

## pooled

### 1. Traces

- n = 30
- truncated (`length`): 8 (26.7%)
- reasoning incomplete (no `</think>`): 6 (20.0%)
- looped: 5 (16.7%)
- accuracy: 21/30 (70.0%)

### 2. Density

- hits per trace: mean 21.63, median 10.50, min 2.00, max 91.00
- hits per 1k reasoning tokens (per trace): mean 8.48, median 7.09, min 2.75, max 22.22
- pooled: 649 hits over 64317 reasoning tokens = 10.09 per 1k

### 3. Per-marker counts

| value | count | share |
|---|---|---|
| `wait` | 421 | 64.9% |
| `hmm` | 79 | 12.2% |
| `let me think` | 52 | 8.0% |
| `hold on` | 36 | 5.5% |
| `no,` | 31 | 4.8% |
| `actually` | 29 | 4.5% |
| `however` | 1 | 0.2% |

**Share of hits that are `wait`: 64.9%** (421/649)

### 4. Position

- paragraph-initial: 307/649 (47.3%)
- mid-token onsets: 342/649 (52.7%)

`prev_token_kind`:

| value | count | share |
|---|---|---|
| `newline` | 307 | 47.3% |
| `word` | 132 | 20.3% |
| `comma` | 96 | 14.8% |
| `period` | 93 | 14.3% |
| `other_punct` | 21 | 3.2% |

### 5. Gaps

Hits with a previous hit in the same trace: 619

| threshold | share of gaps >= | share of first-hit distances >= |
|---|---|---|
| 5 | 83.8% | 100.0% |
| 10 | 77.1% | 100.0% |
| 20 | 68.2% | 96.7% |
| 40 | 57.2% | 76.7% |

First-hit distance from reasoning start: mean 75.83, median 48.50, min 12.00, max 401.00

### 6. Clustering

- hits with another hit earlier in the same paragraph: 192/619 (31.0%)

### 7. Negatives pool

- positions with no hit in the next 100 tokens: **25867** (excluded for insufficient lookahead: 3000)
- positions with no hit in the next 60 tokens: **37153** (excluded for insufficient lookahead: 1800)

By token kind at that position:

| kind | no hit in next 100 | no hit in next 60 |
|---|---|---|
| `newline` | 695 | 933 |
| `period` | 820 | 1166 |
| `comma` | 1593 | 2371 |
| `other_punct` | 2639 | 3588 |
| `digit` | 5001 | 6816 |
| `word` | 13443 | 19875 |
| `other` | 1676 | 2404 |

### 8. Gate input — generated token ids that begin a marker

17 distinct token ids.

| token id | count |
|---|---|
| 14190 | 275 |
| 3783 | 64 |
| 88190 | 53 |
| 13824 | 48 |
| 1077 | 37 |
| 3331 | 35 |
| 8580 | 33 |
| 902 | 31 |
| 3520 | 27 |
| 80022 | 26 |
| 6771 | 11 |
| 10061 | 4 |
| 46936 | 1 |
| 33763 | 1 |
| 52592 | 1 |
| 4354 | 1 |
| 47427 | 1 |

### 9. Paragraph openers the lexicon does not match

Listed only — nothing is added to the lexicon.

| rank | word | n | bigram | n |
|---|---|---|---|---|
| 1 | `so` | 214 | `so, the` | 37 |
| 2 | `but` | 168 | `but wait` | 30 |
| 3 | `alternatively` | 60 | `final answer` | 24 |
| 4 | `therefore` | 55 | `therefore, the` | 19 |
| 5 | `first` | 41 | `so, in` | 19 |
| 6 | `then` | 28 | `alternatively, if` | 18 |
| 7 | `final` | 24 | `but in` | 15 |
| 8 | `1` | 14 | `alternatively, maybe` | 14 |
| 9 | `yes` | 14 | `but let` | 14 |
| 10 | `the` | 12 | `so, if` | 13 |
| 11 | `total` | 12 | `but the` | 13 |
| 12 | `alright` | 12 | `i think` | 11 |
| 13 | `in` | 11 | `alright, so` | 10 |
| 14 | `just` | 10 | `just to` | 10 |
| 15 | `calculating` | 9 | `now, the` | 10 |
| 16 | `another` | 9 | `alternatively, perhaps` | 10 |
| 17 | `okay` | 9 | `so, yeah` | 9 |
| 18 | `since` | 9 | `alternatively, i` | 9 |
| 19 | `fill((0` | 9 | `y` | 9 |
| 20 | `which` | 8 | `first, the` | 8 |
| 21 | `tan` | 8 | `but he` | 7 |
| 22 | `if` | 7 | `yes, that` | 7 |
| 23 | `we` | 7 | `but hold` | 7 |
| 24 | `third` | 7 | `okay, so` | 7 |
| 25 | `given` | 7 | `first, let` | 7 |
| 26 | `divide` | 6 | `given that` | 7 |
| 27 | `next` | 6 | `but that` | 7 |
| 28 | `for` | 6 | `a(1` | 7 |
| 29 | `factor` | 6 | `calculating that` | 6 |
| 30 | `12` | 5 | `another way` | 6 |

### 10. Speed

- tokens/second: mean 63.42, median 63.64, min 61.40, max 64.94


## Answer mismatches (every one, per the review)

| dataset | index | gold | boxed |
|---|---|---|---|
| gsm8k | 201 | `114,200` | `None` |
| gsm8k | 607 | `10` | `1200` |
| gsm8k | 1077 | `145` | `None` |
| gsm8k | 1201 | `42` | `None` |
| math500 | 210 | `10` | `None` |
| math500 | 227 | `\text{(C)}` | `C` |
| math500 | 362 | `\frac14` | `\dfrac{1}{4}` |
| math500 | 372 | `90^\circ` | `None` |
| math500 | 488 | `\$36` | `None` |

## The three-sentence answer

**Do markers fire, and how densely?** They fire readily on both datasets —
649 hits across 30 traces, 10.95 per 1k reasoning tokens on GSM8K (273 hits,
mean 18.2 and median 7.0 per trace) and 9.54 per 1k on MATH500 (376 hits, mean
25.1 and median 19.0) — so the kill condition of fewer than ~1 hit per trace in
both datasets is not remotely met, though the gap between mean and median on
GSM8K is a warning that a few long truncated traces carry much of the total.

**Which markers dominate, and what does the lexicon miss?** `wait` is the
overwhelming majority at 64.9% of pooled hits (68.1% GSM8K, 62.5% MATH500),
with `hmm` (12.2%) and `let me think` (8.0%) the only other markers above 5%;
the most frequent paragraph openers the lexicon does not match are `so` (214),
`but` (168) and `alternatively` (60), of which `but` was deliberately
calibrated out of the FrictionMap lexicon and `alternatively` is the clearest
unclaimed re-evaluation candidate.

**Is GSM8K dense enough to be the primary dataset?** Yes on these counts — 273
onsets from 15 problems, 47.3% of pooled hits paragraph-initial and 83.8% of
within-trace gaps at least 5 tokens, which is the separation the distance bands
need — but the median of 7 hits per trace means a per-problem event budget
should be planned off the median rather than the mean, and the three truncated
GSM8K traces (91, 55 and 36 hits) should be checked for looping before their
onsets are counted as ordinary events.



---

# Addendum — D10 and D11 applied (2026-10-17)

D10: looping traces excluded from all onset-level outputs. D11: onsets
by the paragraph rule — the first hit in a paragraph is the onset, later
hits in that paragraph merge into it.

## D10 exclusions

- traces excluded for looping: **5** (gsm8k_201, gsm8k_1077, math500_151, math500_210, math500_488)
- traces kept: **25**

## D11 — hits to onsets (paragraph rule)

| group | traces | hits | onsets | merged | onsets per trace |
|---|---|---|---|---|---|
| gsm8k | 13 | 146 | **114** | 32 | 8.8 |
| math500 | 12 | 239 | **175** | 64 | 14.6 |
| pooled | 25 | 385 | **289** | 96 | 11.6 |

## Per-marker shares of onsets (paragraph rule)

**gsm8k** (114 onsets)

| value | count | share |
|---|---|---|
| `wait` | 72 | 63.2% |
| `hmm` | 30 | 26.3% |
| `hold on` | 4 | 3.5% |
| `let me think` | 4 | 3.5% |
| `actually` | 3 | 2.6% |
| `no,` | 1 | 0.9% |

**math500** (175 onsets)

| value | count | share |
|---|---|---|
| `wait` | 137 | 78.3% |
| `hmm` | 25 | 14.3% |
| `let me think` | 10 | 5.7% |
| `actually` | 2 | 1.1% |
| `hold on` | 1 | 0.6% |

**pooled** (289 onsets)

| value | count | share |
|---|---|---|
| `wait` | 209 | 72.3% |
| `hmm` | 55 | 19.0% |
| `let me think` | 14 | 4.8% |
| `actually` | 5 | 1.7% |
| `hold on` | 5 | 1.7% |
| `no,` | 1 | 0.3% |

## Window rule (parked — counted once, for the record)

Not used anywhere in this experiment. A hit is an onset unless the
previous hit in the same trace, merged or not, is within N tokens
(difference <= N), per the pre-registration's fixed wording.

| N | onsets, non-looped traces |
|---|---|
| 5 | 324 |
| 10 | 301 |
| 20 | 275 |
