# Question Loop 2: Geometry and Identifiability

**Date:** July 25, 2026  
**Decision:** Replace architecture recovery with cheapest-realization synthesis.

## Does terminal representation geometry contain enough information to prescribe a model?

### Decision

**No. Architecture is fundamentally non-identifiable from terminal geometry.**

This follows from three nested equivalence classes: a Gram matrix has infinitely many feature factorizations; a feature function has infinitely many layer factorizations; and a finite probe set leaves the off-sample function unconstrained.

The Dynamic Geometry Compiler should instead aim to:

> Given a compact specification of a teacher's relational dynamics on a rich probe distribution, synthesize any low-cost student whose dynamics and behavior lie in the same task-relevant equivalence class.

Non-identifiability is then design freedom rather than failure.

## How many representations and architectures share G?

Let `H in R^(N x d)` contain terminal representations and `G = HH^T`. If

$$
G=U_r\Lambda_rU_r^\top
$$

has rank r, every factor

$$
H'=U_r\Lambda_r^{1/2}Q,\qquad QQ^\top=I_r,
$$

has the same Gram. There are continuously infinitely many feature dimensions `d' >= r` and Stiefel factors Q. If G is normalized, centered, or seen only through CKA, scale, translation, and sometimes larger linear equivalences are also unobserved.

There are infinitely many architectures even if the complete terminal function `h(x)`, not just finite H, were known. For

$$
h=f_L\circ\cdots\circ f_1,
$$

insert arbitrary invertible hidden-coordinate maps `phi_r`:

$$
f'_r=\phi_r\circ f_r\circ\phi_{r-1}^{-1}.
$$

The end-to-end function is unchanged while every intermediate representation and parameterization can change. Identity blocks, inverse pairs, neuron splitting, permutations, width expansion, low-rank factorizations, dead branches, and simulations across attention/SSM/RNN substrates create further architectures.

On a finite probe set, add any function `u(x)` that is zero on the probes and arbitrary elsewhere. A lookup table, kernel interpolant, transformer, SSM, RNN, or expressive MLP can agree on H and disagree everywhere not measured. There is no meaningful finite count: compatible models form a continuum across infinitely many graph topologies.

## What additional information is necessary?

For transformation identification *on an observed trajectory*, one needs at least:

- paired states `H_r` and `H_(r+1)` on the same probes;
- a depth clock or reparameterization rule;
- a gauge convention saying when adjacent feature coordinates are the same;
- sufficient rank/persistent excitation; and
- a structural class such as linear, affine, low-degree, or a known operator family.

Without a clock, the same path has arbitrary velocity. Without a gauge, layerwise rotations change the apparent update. Without excitation, unvisited directions are unidentified.

For identification *off the trajectory*, update pairs are insufficient for an unrestricted nonlinear map. One also needs local response information such as

$$
J_r(h)=\frac{\partial f_r(h)}{\partial h},
$$

or finite differences around each state, plus smoothness/model-class assumptions. A finite trajectory cannot determine an arbitrary nonlinear vector field away from its support.

## What A captures that G does not

Let rows index probes and assume a common adjacent feature width and gauge. Define

$$
G_r=H_rH_r^\top,\qquad A_r=\dot H_rH_r^\top.
$$

Then

$$
\dot G_r=A_r+A_r^\top.
$$

Therefore

$$
\operatorname{sym}(A_r)=\dot G_r/2,
\qquad
\Omega_r=\operatorname{skew}(A_r)=(A_r-A_r^\top)/2.
$$

G records current relations. Gram velocity records instantaneous expansion/contraction. Omega records directed circulation of updates relative to the current probe span. For pure feature rotation `dH = HB` with `B^T=-B`, `dG=0` while generally `A=HBH^T != 0`. A therefore contains first-order information absent from an instantaneous Gram or Gram velocity.

## Is A sufficient?

In general, no. The map `dH -> A=dH H^T` has kernel

$$
\{Z:ZH^\top=0\}.
$$

Updates orthogonal to the row span of H are invisible. If H has full column rank, the update is recoverable:

$$
\dot H=AH(H^\top H)^{-1}.
$$

In general,

$$
AG^\dagger H=\dot H\,P_{\operatorname{row}(H)}
$$

recovers only the projected update.

The pair `(G,A)` determines transformation geometry up to a common orthogonal gauge only if:

1. adjacent states share a feature space or declared adapter;
2. updates lie in the observed state span, or H has full column rank;
3. the transition is first-order Markov and linear/affine on that span;
4. probes persistently excite relevant directions;
5. the depth clock is fixed; and
6. the same gauge is used across the transition.

For arbitrary nonlinear layers, A is an empirical tangent statistic, not a unique transformation.

## The hidden gauge axiom

Static Grams are invariant under independent orthogonal changes `H_r -> H_r Q_r`. A raw difference `H_(r+1)-H_r` is meaningful only if `Q_(r+1)=Q_r`, or after choosing an alignment. Transformers, SSMs, and RNNs with a persistent residual/recurrent state provide a natural within-model gauge. Width changes, token pooling changes, and cross-architecture layer matching do not.

This forces a choice:

- If the target is quotient geometry `G(t)`, the full Gram path is already a complete relational description. No additional gauge-invariant rotation exists in that quotient.
- If the target includes transport of feature coordinates, the compiler must supply a connection/gauge. The skew part of A is that extra connection-like information, but it is not assumption-free.

## What “prescribe a model” should mean

The compiler cannot infer “use these 28 transformer layers” from geometry. It can output a control specification

$$
\mathcal S=\{G(t),\mathcal R(t),\text{probe distribution},\text{task boundary}\}
$$

and search a cheap hypothesis class for any student realizing it. The meaningful theorem is:

> If two systems match a specified dynamic relational object on a sufficiently rich probe distribution, under stated regularity assumptions, their task behavior differs by at most epsilon.

That is a realization theorem, not architecture recovery.

The 2026 work of Lutz et al. is instructive: it extracts an explicit depth recursion only after imposing feature- and label-permutation equivariance in a constrained in-context classification setting ([arXiv:2604.11613](https://arxiv.org/abs/2604.11613)). Identifiability comes from strong symmetry and model-class assumptions, not terminal geometry alone.

## NARRATIVE ATTACK

**“That is obvious.”** Weight non-uniqueness is obvious. The stronger result is that a perfect terminal Gram cannot identify the decomposition into computation at all. Architecture prescription from G is an ill-posed inverse problem.

**“That is trivial.”** Replacing recovery with cheapest realization is not semantic retreat. It changes the theorem, objective, search space, and economic promise. Freedom to realize one program on a cheaper substrate is the democratizing opportunity.

## MISSION TEST

Reconstructing a teacher's architecture reproduces its cost structure. Searching the equivalence class for the cheapest realization directly serves cheap intelligence. The program remains aligned only if the dynamic specification is portable, reusable, and adds no student inference overhead.

## What would the result need to BE for the narrative to be unkillable?

A theorem must state its equivalence class, gauge, clock, coverage, and regularity assumptions. Empirically, the same compact specification must produce behaviorally equivalent students on dissimilar substrates, including transformer-to-SSM/RNN transfer, without copying teacher weights. It must outperform architecture search and standard KD at matched total compute. Recovering the teacher's nominal architecture is neither necessary nor sufficient.
