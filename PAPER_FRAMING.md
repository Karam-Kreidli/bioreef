# Paper Framing Notes

Working notes on how to frame the BioReef-Classify paper — the positioning, the
section structure, and the claims each result can and cannot support. Written
from the analysis done while the campaign ran; **verify every number against the
current `RESULTS.md` before it goes into prose** (some early single-seed reads
changed after multi-seed).

Status tags: **[DECIDED]** = settled, **[OPEN]** = still a judgement call,
**[VERIFY]** = check the latest numbers first.

---

## 1. Core positioning — what the paper *is*

**[DECIDED] Lead with the benchmark, not the model.** The paper's two contributions
are (a) a rigorous, leakage-safe benchmark for fine-grained reef-fish
classification on OzFish, and (b) a strong reference method on it. Framing it as
"we propose a benchmark AND a method that sets a strong reference" is far harder
to attack than "our model is #1" (which is weak when you designed the test).

- The benchmark section is a **co-star**, not a preamble — comparable in length to
  the method section, arguably carrying more novelty (good benchmarks are rarer
  than good models).
- **MATANet (C08) is what makes "#1" credible** — it's the closest prior work,
  run from *their* official repo on *our* split. Beating a real external method
  matters far more than beating your own ablations. (C08 still pending a run that
  fits memory — rented A40/L40S/A100.)

## 2. The honest headline claim

**[VERIFY] The defensible, error-bar-safe claim is about mistake *severity*, not
raw accuracy.** At 3 seeds, the proposed model (A15) does NOT clearly win top-1 or
the tail; it wins the *taxonomic-coherence* metrics:

- A15 vs C07 (best timm rival), both 3-seed: A15 wins **HD, mistake-severity,
  genus, family**, and a small **top-1** margin — all outside the error bars.
- A15 **loses the tail** to C07 (C07 ~.521±.018 vs A15 ~.444±.110), and A15's tail
  variance is ~6× C07's. **Do not claim tail supremacy.**

Framing: *"For a taxonomic monitoring deployment, the severity of an error matters
more than raw top-1: our method makes the least taxonomically severe mistakes
(HD, mistake-severity, per-level accuracy) at a small top-1 gain, at the cost of
higher variance on the rare tail."* This is the strongest claim the data supports.

## 3. The metric panel is itself an argument

**[DECIDED]** The plain-CE result is the concrete proof that "top-1 isn't enough."
Plain CE (A14) posts the table-best top-1 but makes structurally worse mistakes
(higher HD / mistake-severity). Use this to *motivate* the metric panel (HD,
mistake-severity, head/medium/tail, per-level) rather than treating it as an
embarrassment. A reviewer who sees you report a result that superficially beats
your model, then explain why the deeper metrics matter, trusts the whole paper more.

## 4. Section structure

**[DECIDED]** Suggested order and relative weight:

1. **Intro** — reef monitoring, why fine-grained + taxonomy-aware, why existing
   evals leak.
2. **Benchmark / Dataset** (co-star, full length) — see §5.
3. **Method** (co-star, full length) — MCEAM, HSLM, CB-Focal, frozen-vs-unfrozen
   backbone. Every design choice traced to a dataset property (§6).
4. **Experiments** — baselines (C01, C03–C07 timm, C08 MATANet), the proposed
   model, protocol.
5. **Ablations** — as controlled one-variable studies (§7).
6. **Limitations + Discussion** — state them plainly (§8).

Method and Benchmark sections roughly equal length. Do NOT starve the benchmark
section to inflate the method — a rigorous split section is what makes this a
benchmark contribution rather than "we trained a model."

## 5. The dataset/split section — what to cover

**[DECIDED]** This is where rigor is demonstrated. Cover, in order:

1. **The leakage problem** (lead with it): OzFish crops are BRUVS stereo-deployment
   frames; the same individual recurs across consecutive frames AND both `_L`/`_R`
   cameras. A crop-level random shuffle leaks identity → inflated test accuracy.
   *If possible, quantify it* — one number with a naive random split vs the grouped
   split; that accuracy gap is worth a figure.
2. **Grouping unit = deployment** (not crop/frame/camera). State the invariant the
   code *checks*: deployments are disjoint across folds (`_log_split_summary`
   verifies + errors on leak).
3. **Stratification**: rarest-species-first greedy assignment — best-effort
   per-species fold coverage subject to the hard grouping constraint.
4. **Inclusion rules = the benchmark definition**: ≥20 crops AND ≥3 deployments
   (the ≥3 is leakage-motivated — guarantees splittability), placeholders dropped.
   Report: **321 species, 39 families, 127 genera, ~76,083 crops** at seed 0.
5. **Class-key discipline** (a real, citable subtlety): the class key is the full
   (genus, species) binomial, never the bare epithet — **49 epithets in OzFish are
   shared across genera** (e.g. `niger`), which a bare key silently collapses into
   one class. Plus genus-typo canonicalization and the `Epinephelus faveatus`
   family correction. Shows you *audited* the labels.
6. **Reproducibility artifact**: the released split file, one row per annotation
   keyed by `file_name + bbox` (frame alone isn't unique — 22,223 seed-0 rows
   share a file_name). Deterministic across machines/processes (sorted-before-
   shuffle, name tie-breaks removed hash-seed nondeterminism).
7. **Limitations, stated not hidden**: unsplittable rare species absent from
   some folds (report the count); realized ratios drift from 70/15/15 (report the
   actual %); `strict_images` = every referenced image must be present.

**[VERIFY]** Numbers 4, 5, 6 need real figures from the seed-0 split log — a small
script running `_log_split_summary` can dump them into a table.

## 6. Design choices trace to dataset properties

**[DECIDED]** Present these as motivated, not a bag of tricks:

| dataset property | design response | ablation |
|---|---|---|
| long tail | CB-Focal (+ tested balanced sampling, which LOST) | A7 |
| coarse-to-fine confusability | HSLM hierarchy loss | A6/A13/A15 vs A12 |
| context-dependent reef ID | MCEAM environmental attention | A2–A5 |
| small deployable footprint | frozen backbone (+ unfreeze sweep) | A9–A12 |

Keep the split section (fixes *evaluation* problems: leakage, label collapse) and
the method section (fixes *task* problems: tail, hierarchy, context) DISTINCT.
Cross-reference but don't conflate — the split makes the benchmark honest, the
method performs on it.

## 7. Ablation framing

**[DECIDED]** Frame as controlled one-variable studies off the reference config
(the C09 discipline pays off here). Two moves that strengthen it:

1. **Report sweeps as curves, not single wins.** The unfreeze sweep
   C09→A9→A10→A11 is a *frontier* ("how much does adaptation depth buy"), and the
   A11→A12 LR probe is a separate finding ("the panel LR was too hot for full FT").
2. **Report where components lose.** Balanced sampling (A7) lost. CB-Focal's tail
   gain is within noise on the strong backbone (A14→A13). Reporting these builds
   credibility.

**[DECIDED] Reference model repositioned:** the frozen C09 is the *frozen baseline /
start of the unfreeze frontier*, NOT the headline. The proposed model is the
unfrozen config. Frozen loss ablations (A6/A8) are demoted to a one-paragraph
"the loss barely separates on a frozen backbone" justification for re-cutting the
loss ablation unfrozen (A13/A14/A15) — keep their result JSONs as that evidence.

**[VERIFY] The loss 2×2 (unfrozen):** A14 (CE) / A13 (CBF) / A15 (HSLM+CE) /
A12 (HSLM+CBF). Conclusions to confirm against final numbers:
- **HSLM helps regardless of species loss** (A14→A15 and A13→A12 both improve
  HD / mistake-severity / per-level) — the strong, clean loss finding.
- **CB-Focal's tail benefit is within noise** on the strong backbone while it
  costs top-1/HD → A15 (HSLM + plain CE) is the leaner proposed model.

## 8. Limitations to state plainly

**[DECIDED]** Stating these *builds* credibility:

- Single-benchmark, single-dataset (OzFish); geographic/gear specificity of BRUVS.
- The proposed model's **tail is weak and high-variance** — honestly the biggest
  caveat. C07 (plain Swin) beats it on the tail.
- Unsplittable rare species (some absent from val/test folds).
- The proposed config's LR differs from the frozen reference — it's the best
  *found* config, reported as such, not a single-field ablation of the frozen ref.
- C08/MATANet required more memory than a 24GB card provides at its published
  config (a fair point about the practical cost of a 4× fine-tuned ViT-large
  vs. our approach) — see §9.

## 9. Backbone finding (resolved)

**[DECIDED, VERIFY numbers]** See memory `dinov2-beats-dinov3-bioreef`:
- **Frozen: DINOv2 > DINOv3** (A1 > C09, 3 seeds, clean). Report it — the newer SSL
  backbone transfers *worse* frozen to this narrow domain.
- **Unfrozen: DINOv2 ≈ DINOv3** (A16 ≈ A15, a statistical tie once fine-tuned).
  The single-seed DINOv2 "win" did NOT survive 3 seeds.
- Framing: *"backbone generation matters frozen but washes out under fine-tuning."*
  Report the proposed model on DINOv3 for lineage continuity; note DINOv2 is
  equivalent unfrozen and better frozen. **No backbone switch needed.**
- **Do NOT** run "DINOv2 + CB-Focal unfrozen" — settled as unnecessary.

## 10. The paper↔deployment through-line

**[DECIDED]** The campaign doubles as model selection for the Junior deployment
(35-species local reef data). The best paper config gets rebuilt there via
transfer learning from the OzFish weights — never trained from scratch (the
frozen self-supervised backbone is the asset that makes small-data feasible).
- Two candidates carry forward: the accuracy/coherence winner (A15/A12) and a
  tail-hedge (A11, or a tail-favoring variant), because the deployment may weight
  the rare tail differently than the paper's headline.
- Deployment backbone choice isn't forced by accuracy (unfrozen tie) — if the
  deployment must freeze (cheaper/cacheable), DINOv2 wins; if it fine-tunes,
  either works.

---

## Open decisions to make before/while writing

- **[DECIDED 2026-08-17]** ~~Is the proposed model A15 or A12?~~ **A15 (HSLM + plain
  CE, unfrozen DINOv3) is the proposed model.** CB-Focal is dropped — it doesn't earn
  its place on the strong backbone (its tail gain is within noise while it costs
  top-1/HD; A13→A12 vs A14→A15). C09 is repositioned as the *frozen baseline*, not
  the proposed model. This is now reflected in C09_proposed.yaml, MANIFEST.md.
- **[OPEN]** Do we run A2-unfrozen (MCEAM on/off on the strong backbone)? Optional
  rigor; only if a reviewer would demand the context ablation on the final backbone.
- **[OPEN]** How much to feature the frozen results at all, now that the proposed
  model is unfrozen.

---

## 11. Decisions & results logged 2026-08-17 (post-audit + C08 first number)

**[DECIDED] Proposed model is A15** — see the resolved [DECIDED] above. All docs
(configs, MANIFEST) now say A15=proposed, C09=frozen baseline. **In the paper, do
NOT narrate the post-hoc chronology** ("we added A12 after seeing A11"): the repo is
uploaded fresh, all configs present at once, so every config reads as an *initial*
design choice. Present the ablation grid as a designed panel, not a timeline.

**[DECIDED] Benchmark coverage is "321 defined / 313 evaluable."** The inclusion
rule (≥20 crops, ≥3 deployments) defines **321 species**, but the deployment-grouped
split does not place every species into every fold: test covers **313** (val 315;
seeds 1/2 → 312/311). `macro_accuracy` averages over classes *present in test*, so
the headline macro is over 313, not 321. **Report this explicitly** — "321-species
benchmark; 313 evaluable in the seed-0 test fold" — as a stated limitation (§5.7),
NOT a re-split. This is honest and defensible for a leakage-safe benchmark; hiding
it is the only way it becomes a problem.

**[VERIFY→confirmed] C08/MATANet, seed 0 (first real number, L40S):**
- top1 **0.841**, macro **0.705**, HD **0.283**, mistake-sev **1.783**,
  family 0.953 / genus 0.923, tail 0.485.
- **Provisional read (1 seed of C08 vs 3 of A15):** C08 ≈ ties A15 on top1
  (0.841 vs 0.842±.006) and is marginally better on HD/mistake-sev — INSIDE the
  noise band. **Do not claim A15 beats C08 yet; need C08 seeds 1/2.**
- **The headline framing this unlocks — parameter efficiency.** C08 is a ~1 B-param,
  4-encoder fine-tuned DINOv2-large; A15 is a fine-tuned DINOv3-base (~86 M backbone).
  If A15 matches C08's coherence metrics at ~12× fewer backbone params, the claim is
  *"we match the closest heavy prior work at a fraction of the trainable footprint,"*
  which is stronger than a raw-accuracy win. Both are unfrozen, so this is NOT a
  frozen-vs-heavy story — it's efficiency + taxonomic coherence.

**[DECIDED] MATANet comparison wording (audit #64/#65).** Say MATANet is given the
**identical taxonomy tree** and is **scored by our identical evaluation harness** —
NOT "identical hierarchical supervision." Our HSLM marginalises species→parents (one
head); MATANet uses native per-level heads (`lambda_sub_h`/`lambda_ce`). And our eval
HD is 0/1/2/3 severity while MATANet's *training* tree-distance matrix is 0/2/4/6 —
different training objectives, common evaluation. State it that way; it's both honest
and still a fair comparison.

**[DONE] DDP path removed.** The multi-GPU `scripts/train.py`/`test.py` was never
used (all results came from single-GPU `run.py`); it's deleted, not fixed. So the
audit's DDP bugs (wrong loss wiring, LR×world_size, etc.) never touched any reported
number. One trainer, one code path — simpler reproducibility story for the paper.

**[POINTER] Full audit tracker: `AUDIT.md`** — 86 findings with status. Reproducibility
work still pending before release (split-manifest checksums, provenance hashing,
aggregator `--campaign` filter to exclude D1/D2 + keep the table to declared runs).
The benchmark is the co-star contribution (§1), so this hygiene matters.
