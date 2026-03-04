# COLM 2026 Restructuring Plan

## Constraint
- **Main text: 9 pages** (strict)
- **References: unlimited** (after bibliography)
- **Appendix: unlimited** (after bibliography)
- **Reproducibility/Ethics/Acknowledgments: 1 page each** (not counted)
- **Double-blind**: use `\usepackage[submission]{colm2026_conference}`, anonymize author

## Current State
- Paper: 31 pages (main ~20 pages, appendix ~8 pages, bibliography ~2 pages)
- Need to cut ~11 pages from main text

## Restructuring Strategy

### 9-Page Main Text

**Page 1: Title + Abstract (0.7 pages)**
- Keep abstract as-is (already tight at ~0.7 pages)
- Remove line numbers for camera-ready

**Pages 1-2: Introduction (0.8 pages)**
- Current intro is already compact. Keep as-is.

**Page 2: Related Work (0.5 pages)**
- Compress to 3 paragraphs: (1) scaling laws, (2) representation quality, (3) EVT + neural collapse
- Move the generation law related work (Wu&Papyan, Kulkarni, Yang) to appendix

**Pages 2-3.5: Theory (1.5 pages)**
- Keep definitions, Gumbel race, main theorem
- Move hierarchical beta derivation details to appendix
- Move connection-to-theory paragraph to appendix (alpha-rho, equicorrelation details)

**Page 3.5-4: Experimental Design (0.5 pages)**
- Keep architecture list, pre-registration paragraph
- Remove cross-modal experiment details (move to appendix)

**Pages 4-7.5: Results (3.5 pages)**

*4.1 LOAO (1 page): KEEP Table 1 + key text*
- Single-C0: alpha=2.866, CV=0.019
- Per-dataset: alpha=1.477, CV=0.023, R^2=0.955
- RWKV boundary test: 1 paragraph (move details to appendix)
- Figure 1 (main law plot)

*4.2 Cross-modal + OOD (0.7 pages)*
- ViT R^2=0.964 (1 sentence)
- SmolLM2 blind OOD r=0.817 (1 paragraph)
- H8+ holdout: all 6 PASS, r=0.879, MAE=0.077 (1 paragraph)
- Move H8+ per-model table to appendix
- Keep Figure 2 (H8+ scatter)

*4.3 Causal evidence (0.5 pages)*
- Keep causal summary table (Table 2)
- Confusion-matrix: r=0.842-0.776, sign acc 93-100% (1 paragraph)
- Move competition field details, surgery, weight maps to appendix

*4.4 Biological validation (0.5 pages)*
- Allen Neuropixels: 30/32 PASS, mean r=0.736 (1 paragraph)
- Multi-area: all 5 areas PASS (1 sentence)
- Equicorrelation: rho~0.46, CV=1.65% (1 sentence)
- Keep Figure 3 (Allen biological)

*4.5 Generation law (0.5 pages)*
- 22 models, r=-0.70, fixed-V r=-0.84 (1 paragraph)
- K_eff decomposition (1 sentence)
- Move model table to appendix
- Keep Figure 4 (generation scatter)

*4.6 Three-level universality + alpha-rho (0.3 pages)*
- Three-level structure (1 paragraph)
- Alpha-rho: +4.3% error, zero params (1 sentence)
- Alpha noise: CV=2.3% = estimation noise (1 sentence)

**Pages 7.5-8.5: Discussion (1 page)**
- What is universal (3 bullet points)
- Honest scope (2 bullet points)
- Connection to physics (1 sentence)
- Compress current ~2 pages to 1

**Pages 8.5-9: Limitations + Conclusion (0.5 pages)**
- Top 4 limitations only: (1) cross-task drift, (2) external replication needed, (3) large-K attenuation, (4) assumption dependence
- Conclusion: 1 paragraph

### Appendix Structure

**Appendix A: Derivation Details** (current)
**Appendix B: Pre-registration Deviation Log** (current)
**Appendix C: Extended Results**
- RWKV boundary test details
- H8+ per-model table
- Cross-modal full results (ViT-Base, ResNet50, K=100 analysis)
- K-scaling and one-point calibration
- Comprehensive 19-architecture universality
- Sparse competition analysis
**Appendix D: Causal Evidence**
- Surgery details
- Competition field full analysis
- Two-component geometry
- Confusion matrix Gumbel test
- Weight map details
**Appendix E: Biological Validation**
- Multi-area batch details
- Equicorrelation details
- Cadieu/Stringer cross-species
**Appendix F: Generation Law**
- Model table
- Hypothesis scorecard
- Proxy B analysis
**Appendix G: Alpha-Rho Prediction**
- Full alpha-rho derivation and validation

### Figures in Main Text (4 figures)
1. `fig_cti_universal_law.png` — main LOAO plot + architecture stability
2. `fig_cti_h8plus_holdout.png` — H8+ expanded holdout
3. `fig_cti_allen_biological.png` — Allen Neuropixels biological
4. `fig_generation_law.png` — Generation law scatter

### Tables in Main Text (3 tables)
1. LOAO 12-architecture table (current Table 1)
2. Causal summary table (current Table 2)
3. Cross-modal results table (current Table 3)

### Key Sections to Remove from Main Text
1. Competition field (Section 5.6) → Appendix D
2. K-scaling (Section 5.7-5.8) → Appendix C
3. Composite kappa-K sweeps (Section 5.9) → Appendix C
4. Sparse competition (Section 5.10) → Appendix C
5. Beta interpretation paragraph → Appendix C
6. Scaling dynamics paragraph → Appendix C
7. Extended limitations 5-12 → Appendix

## Estimated Effort
- Main text restructuring: 2-3 hours of careful editing
- Appendix reorganization: 1-2 hours
- Total: ~4-5 hours

## Needs Codex Review Before Implementation
- Which results are most important for the 9-page main text?
- Should H3 ranking table stay in main text or move to appendix?
- Should multimodal summary figure replace H8+ figure in main text?
- How to handle the Discussion section compression?
