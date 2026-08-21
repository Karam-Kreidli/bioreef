# Deployment Optimization Notes (NOT for the paper)

Personal / project notes on squeezing the best possible model out of the ablation
results, for the **Junior deployment** (35-species local reef). This is beyond
the paper — the paper reports the clean campaign; this is "given everything we
learned, what's the strongest model we can actually build."

Baseline to improve on: **A16** (the top run): DINOv2-base, unfreeze all 12 blocks
(full FT), lr 1e-5, MCEAM 3-context / attention_depth 1, HSLM + plain CE, random
sampler. 3-seed: HD .280, top1 .846, macro .692, tail .453±.114.

## Why there isn't much low-hanging fruit

A16 is already the accumulation of **every winning ablation choice**. The knobs
and where A16 sits:

| knob | ablation result | A16 | headroom? |
|---|---|---|---|
| unfreeze depth | monotonic C09→A9→A10→A11→A12, huge effect | full (12) | **maxed** |
| backbone | DINOv2≈DINOv3 unfrozen (tie) | DINOv2 | at optimum |
| LR | 1e-4→1e-5 big win for full FT | 1e-5 | corrected |
| HSLM | A14→A15 clear help | yes | keep |
| MCEAM context on/off | A2 (off) catastrophic (.557 top1) | 3-context | essential, keep |
| MCEAM ROI scale | A3 ≈ ref, negligible | default | none |
| sampler | A7 (balanced) lost badly | random | keep |
| CB-Focal | tail gain within noise on strong backbone | dropped | see tail below |
| **MCEAM attn depth** | A4/A5 (depth 2/4) barely moved — **but only tested FROZEN** | depth 1 | **UNTESTED unfrozen** |

The single biggest lever (unfreeze depth) is already maxed, which is why A16 is on
top. Most remaining knobs have no headroom. Real candidates below.

## Candidates to try, ranked by expected payoff

### 1. MCEAM attention_depth 1 → 2 (or 4) on the UNFROZEN backbone  ★ best bet
A4 (depth 2) and A5 (depth 4) barely beat depth 1 — **but they were run on the
frozen C09 backbone**, where the backbone was the bottleneck and the attention
module couldn't show its value (exactly why the frozen *loss* ablations were also
inconclusive). On the strong unfrozen backbone, deeper cross-attention over the
context streams may finally pay off. One-field change off A16.
- Run: A16 config + `attention_depth: 2` (and maybe a `: 4`).
- Cheap, high-information — this is the one dimension genuinely untested on the
  regime that matters.

### 2. Bring CB-Focal back on top of HSLM (= A12's recipe on DINOv2)  ★ for the tail
For deployment the **tail matters** (rare local species), and A16's tail is its
weak spot (.453±.114, high variance). A12 (HSLM + CB-Focal, unfrozen DINOv3) had
the best/most-stable tail of the unfrozen runs (.497±.034). So "DINOv2 + HSLM +
CB-Focal, unfrozen" may trade a hair of top-1/HD for a better, steadier tail —
which for deployment can be the right trade.
- NOTE: this is the "DINOv2 + focal" run explicitly skipped FOR THE PAPER (settled
  as unnecessary there). It's relevant HERE because deployment weights the tail
  differently. Different objective, different answer.
- Run: A16 config + `loss: cbfocal` (keep hslm: true).

### 3. Longer training / EMA sweep  (untested dimension)
No ablation swept epochs or EMA decay. A cooler LR (1e-5) converges slower, so
full FT might still be improving at epoch 60. Also check whether the
best-by-val-HD checkpoint is landing near the end (under-trained signal).
- Try: epochs 90–120 on the A16 config; watch the val-HD curve for a plateau.
- Possible free gain, no config-space risk.

### 4. Ensemble A15 (DINOv3) + A16 (DINOv2)  (deployment-only)
The two backbones are a statistical tie but likely make *different* errors, so a
2-model ensemble (average logits) could beat either. Heavier at inference, but for
a local system where accuracy > latency it's a legitimate option. Also naturally
reduces the tail variance that plagues both single models.

### 5. Tail-targeted tricks not in the ablation space  (deployment-only, exploratory)
The ablations never tried: logit-adjusted loss (Menon et al.), two-stage decoupled
training (cRT / LWS — train backbone with instance sampling, then re-train the head
with class-balanced sampling), or test-time logit adjustment. These are the
standard long-tail toolkit and directly target A16's weakness. Out of scope for
the paper's one-variable discipline, but exactly what a *deployment* model should
consider.

## Recommended order for the deployment build

1. **A16 + attention_depth 2** — the one untested lever in the regime that matters.
2. **A16 + CB-Focal (A12-recipe on DINOv2)** — for the tail, if rare species matter.
3. Compare those three (A16, A16+depth2, A16+CBF) on the tail specifically.
4. If the tail is still the blocker, go to decoupled training / logit adjustment (#5).
5. Ensemble (#4) as the final accuracy squeeze if inference cost allows.

## Important reminder for the actual deployment

When this moves to Junior's 35-species local data, the winner is rebuilt via
**transfer learning from the OzFish weights** — never trained from scratch (the
self-supervised backbone is the asset that makes small-data feasible). Re-validate
the ranking on the local set: "best on OzFish (321 sp)" should transfer, but
confirm it on 35 species with less data before locking it in. See
[[paper-to-deployment]] framing and `PAPER_FRAMING.md` §10.
