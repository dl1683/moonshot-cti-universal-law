## Decision

Fund exactly one analytic Gate-0 day—but do not continue with the weak-feature SQ selector in Sections 9–10.

The live workbook has already superseded that route: the scalar weak-list uniformization conjecture is refuted by decision lists and halfspaces using Daniely–Feldman. The surviving object is the **strong marginal selector that commits to an entire \(d\)-dimensional representation before labels**. See the [weak-route refutation](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/research/AI_PARADIGM_BREAKTHROUGH_WORKBOOK.md:6491>) and the [current Gate-0 fork](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/research/AI_PARADIGM_BREAKTHROUGH_WORKBOOK.md:7645>).

## The first lemma

Attempt the **sfat-1 Robust Transcript-Cover Lemma**.

Let \(A\) be a finite deterministic, target-blind marginal selector of arbitrary depth \(q\), tolerance \(\tau\), and let \(\mathcal U_A\) contain every marginal query reachable in its tree. Define the dual answer class

\[
\mathcal F_A
=
\left\{
f_D:\mathcal U_A\to[-1,1],
\quad
f_D(u)=\mathbb E_D[u]
:
D\in\Delta(X)
\right\}.
\]

For each rounded transcript \(t\), let \(C_t\) be the set of marginals for which \(t\) is tolerance-valid. Prove:

> If \(\operatorname{sfat}_{\tau/8}(\mathcal F_A)\le 1\), then there is a distribution \(\lambda\) over actual transcripts of \(A\) such that
> \[
> \inf_D \lambda\{t:D\in C_t\}\ge \tau^{O(1)},
> \]
> independently of \(q\).

If every valid transcript supplies a \(d\)-dimensional randomized representation supporting every \(h\in H\) at error \(\epsilon\), transcript sampling and block concatenation would then give

\[
\operatorname{pdc}_{C\epsilon,\delta}(H)
\le
d\,\tau^{-O(1)}\log(1/\delta),
\]

again independent of \(q\). Only after proving the sfat-1 case should the recurrence be generalized toward

\[
\rho_A^*\ge(c\tau)^{O(r)}
\quad\text{for}\quad
r=\operatorname{sfat}_{c\tau}(\mathcal F_A).
\]

Why this lemma:

- It permits arbitrarily many marginal queries, so it is not the already-known constant-query enumeration result.
- It uses the adaptive geometry of the selector’s marginal-answer class—structure an arbitrary VC class does not possess.
- It attacks the exact unresolved technical gap: existing sequential fractional covers concern pathwise approximation of prediction trees, whereas we need a distribution over **self-consistent, coordinatewise tolerance-valid transcripts whose path is induced by its own answers**.
- If this fails already at sequential fat dimension one, there is no credible reason to believe unrestricted SQ-selector structure will make FKS/PDC easier.

Do not begin with “mixture forces compatibility.” That is too close to attempting the entire open problem, and the workbook already explains why convex transcript cells and naive marginal mixing do not glue coupled targets.

## One-day kill test

The clock starts after the reading packet below.

**Stop Gate 0 after the day if:**

- the sequential fractional cover cannot be converted into actual endogenous, \(\ell_\infty\)-valid oracle transcripts;
- the proof must assume \(\rho_A^*\), low selector-relative cover rank, polynomially many routes, or a common predictor space—i.e. it assumes the conclusion;
- the only bound is \(O((1/\tau)^q)\);
- or the result is completely absorbed by an existing sequential-cover theorem and creates no new selector-to-PDC consequence.

An unresolved self-consistency gap at day’s end counts as failure, not “promising.”

**Continue only if:**

- there is a complete finite proof of the sfat-1 statement with adversarial tolerance semantics and no dependence on \(D\), \(h\), or \(q\);
- the proof correctly handles selector and representation randomness, or clearly isolates the deterministic lemma needed for that extension;
- it passes both calibrations:
  - one-parameter moment selectors have bounded sequential answer dimension and collapse polynomially;
  - binary-search point-mass selectors have sequential dimension \(\Theta(q)\), correctly recovering exponential fragmentation rather than falsely collapsing it.

That result would pass Gate 0. It would not solve FKS, but it would establish a genuine selector-specific boundary worth developing.

## Mandatory reading

Read these exact results, in this order:

1. **Feldman–Kamath–Srebro (2026)**, the complete six-page invited open problem. This fixes the frontier we must not rename: [PMLR 336](https://proceedings.mlr.press/v336/feldman26a.html).

2. **Kamath–Montasser–Srebro (2020), arXiv:2003.04180**: Definitions 1–2, Theorems 6–8, and Appendix B.2. This fixes common versus distribution-dependent PDC and the unresolved PDC-versus-VC question: [paper](https://arxiv.org/abs/2003.04180).

3. **Block–Dagan–Rakhlin, arXiv:2102.01729**: Definition 6, Theorem 13, Proposition 14, and Lemmas 23–24. This is the closest existing machinery: fractional sequential covers bounded by sequential fat-shattering dimension. The missing step is the valid-transcript transfer described above: [paper](https://arxiv.org/abs/2102.01729).

4. **Rakhlin–Sridharan–Tewari (2015)**, “Sequential Complexities and Uniform Martingale Laws of Large Numbers,” especially the sequential fat-shattering definition and sequential Sauer–Shelah/cover bounds: [DOI](https://doi.org/10.1007/s00440-013-0545-5).

5. **Daniely–Feldman, arXiv:1809.09165**, Lemma 2.6 and Theorem 3.1. This prevents accidentally resurrecting the already-dead scalar weak-list route: [paper](https://arxiv.org/abs/1809.09165).

6. **Feldman, arXiv:1608.02198**, the distribution-independent randomized SQ characterization. Audit whether the proposed answer-class dimension is already implicit in established SQ statistical dimension: [paper](https://arxiv.org/abs/1608.02198).

7. As lower-bound guardrails, read **Alon–Moran–Yehudayoff, arXiv:1503.07648** and the 2026 approximate-sign-rank result **arXiv:2605.01038**. They explain why large exact or deterministic approximate sign rank is not the required common-PDC obstruction: [AMY](https://arxiv.org/abs/1503.07648), [Bindu–Hatami–Karimi–Robere](https://arxiv.org/abs/2605.01038).

## Repository decision

This work does **not** belong in `moonshot-cti-universal-law`. That repo remains archived.

Gate 0 belongs exclusively in the portfolio-level [AI Paradigm Breakthrough Workbook](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/research/AI_PARADIGM_BREAKTHROUGH_WORKBOOK.md:1>), whose own contract says not to create parallel investigation notes. Record the definition, proof attempt, and kill/pass result there.

Create a new child repository only after this lemma passes and the strong selector game is frozen. If it fails, record the kill in the workbook and create nothing.

The repeatable story is: **“If an environment can answer only a shallow kind of adaptive question, its many local representations may be forced to collapse into one common representation.”**