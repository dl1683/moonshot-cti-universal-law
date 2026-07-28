# W-D3 (PolicyBench) Analysis Notes — Jul 28, 2026

## Raw Results (3/9 systems complete)

| System | Tasks | Pass | Pass% | Mean Score | vs All-Zero |
|--------|-------|------|-------|------------|-------------|
| **ALL-ZERO BASELINE** | 100 | **100** | **100.0%** | **0.8485** | -- |
| qwen3_0.6b | 100 | 94 | 94.0% | 0.7654 | -8.3pt |
| qwen3_14b | 23 | 19 | 82.6% | 0.7312 | -11.7pt |
| qwen3_4b | 100 | 55 | 55.0% | 0.5096 | -33.9pt |

## Critical Finding: All-Zero Dominance

The trivial all-zero baseline (predict 0 for every field) achieves **100% pass rate
and 84.85% mean household score**. Every model tested so far performs WORSE than this
baseline.

### Why zeros dominate

Gold distribution across 1970 fields (100 households):
- Binary fields (970): 871 zero (89.8%), 99 one (10.2%)
- Numeric fields (1000): 781 zero (78.1%), 219 nonzero (21.9%)
- Overall: 83.9% of all gold values are zero

Most households have no income tax liability, no refundable credits, and are
ineligible for most programs. Predicting all-zero is the majority-class baseline.

### Model behavior profiles

**qwen3_0.6b (94% pass, mean 0.7654):**
- Predicts 0 for 97.2% of numeric fields, 87.2% of boolean fields
- 87 of 95 valid households had ALL predicted values = 0
- Essentially IS the all-zero baseline with parse failures costing 6 points
- Accuracy on nonzero numeric: 4.6% (near zero)
- Accuracy on binary=1: 0% (predicts 0 for all)

**qwen3_4b (55% pass, mean 0.5096):**
- Predicts 0 for 62.6% of numeric fields, 38.9% of boolean fields
- Only 16 of 91 valid households had all-zero predictions
- Attempts real computation but hallucates PLACEHOLDER values (1234.56, 12345.67, 123456.78)
- Over-predicts eligibility: 61.1% of boolean predictions are 1 (gold: 10.2%)
- Accuracy on nonzero numeric: 6.5% (slightly better than 0.6b's 4.6%)
- Accuracy on binary=1: 60% (but with 55.2% false positive rate on binary=0)

**qwen3_14b (82.6% pass, mean 0.7312, partial 23/100):**
- Between 0.6b and 4b in zero-prediction rate
- Higher accuracy on binary=1: 78.9%
- Higher accuracy on binary=0: 89.4%
- Still near-zero on nonzero numeric: 3.8%
- U-shaped: recovers from 4b's degradation

### The "scale inversion" mechanism

The apparent scale inversion (0.6b > 4b) is NOT the small model understanding
tax policy better. It is:

1. The gold distribution is 84% zeros
2. The 0.6b model defaults to zeros (too small to attempt computation)
3. The 4b model is large enough to attempt computation but fails (hallucinated placeholders)
4. The 14b model recovers because it gets binary eligibility mostly right

**The inversion is between "knowing nothing" and "knowing a little" on a
zero-dominated task.** A model that cannot compute at all beats one that
tries and fails, because the base rate is overwhelmingly zero.

### Implications for the Atlas

1. **W-D3 pass rate is not discriminative at the top.** All-zero gets 100%.
   The interesting signal is in the SCORE (mean household accuracy), not pass/fail.

2. **The all-zero baseline MUST be included** as a reference point in Gate A.
   Any model scoring below 84.85% mean is worse than predicting nothing.

3. **Cross-workload divergence is real but trivially explained.** W-D2 (MKQA)
   is discriminative because it requires real multilingual QA. W-D3 has a
   degenerate gold distribution.

4. **W-D3 may need redesign** — or the analysis should focus on the hard subset
   (households with nonzero tax liability, eligible for programs). Currently,
   54 of 100 households have only 16 fields each (simplest case), inflating
   scores for zero-predictors.

5. **The narrative "stop paying for the biggest AI" needs honest grounding.**
   On W-D3, the cheapest AI is "predict zero" — a regex, not even a model.
   The Atlas signal on W-D3 comes from WHICH model first beats the all-zero
   baseline with real computation.

## Pre-Registration (before seeing remaining 6 systems)

H-WD3-1: gemma3 and falcon_h1 families will show the same U-shape as qwen3
(smallest model closest to all-zero, middle model worst, largest recovers).

H-WD3-2: No local model will exceed the all-zero mean score of 0.8485.

H-WD3-3: Parse failure rate will correlate with household complexity
(more fields = longer required JSON = more likely to hit 384 token limit).

H-WD3-4: The 0.5B models (qwen3_0.6b, falcon_h1_0.5b) will have the
highest pass rates because they most closely approximate the zero baseline.
