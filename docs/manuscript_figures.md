# Manuscript figure inputs

Run the scripts from the repository root with Python 3.10 or newer. Paths in the
commands point to local copies of the saved evaluation files. Download or
prepare the inputs for each figure before running its command.

## Figures 3 and 4 · CASP16

```bash
python scripts/plot_casp16_fig3_fig4.py \
  --lddt-summary evaluation/summary_macro_target_metrics.csv \
  --tm-summary evaluation/target_macro_metrics.csv \
  --lddt-top1 evaluation/lDDT_top1_per_target.csv \
  --tm-top1 evaluation/TM_top1_per_target.csv \
  --output-dir figures
```

To use Arial on a machine without system-wide Arial fonts, pass both the
regular and bold font files as separate `--font-file` arguments.

- Each metric summary has `method`, `pearson`, and `spearman` columns. When an
  lDDT summary contains multiple scopes, the script uses
  `common_12_methods_52targets`. The lDDT summary comes from residue-level
  correlations computed within each decoy, averaged within targets and then
  averaged over targets. The TM-score summary averages correlations computed
  across decoys within each target.
- Each Top-1 table has `method`, `target`, `n_decoys`, and
  `selected_true_quality` columns. Supplementary Data S3 and S4 provide the
  per-decoy predictions and reference scores for lDDT and TM-score,
  respectively. Within each target and method, select the candidate with the
  highest prediction, using the first decoy ID in lexicographic order for
  exact ties, and carry its reference score into the Top-1 table.
- The lDDT and TM-score evaluations both cover 52 targets. The common candidate
  counts are 17,742 for lDDT and 17,539 for TM-score. For lDDT selection, a
  decoy's score is its mean predicted or reference lDDT over the evaluated
  residues. Reference TM-score is the arithmetic mean of the official scores
  for the assessed structural domains.
- Each Top-1 bar is the unweighted mean of the reference scores of the selected
  models across targets. Arrow annotations show differences relative to
  DeepAccNet.

## Figure 5 · CASP14 Stage-2 comparison

```bash
python scripts/plot_casp14_fig5.py \
  --per-target evaluation/Fig5_17_methods_per_target_without_T1045s1.csv \
  --roc-curves evaluation/stage2_macro_roc_curves_exclude_T1045s1.csv \
  --output figures/Figure5_CASP14.png
```

- The per-target table has `display_name`, `method`, `source`, `target`,
  `multidomain`, `pearson`, `spearman`, `chosen_true_lddt`,
  `best_true_lddt`, and `topone_loss`. Set `source` to `ours` for
  SaProt + DeepAccNet, ESM3, and DeepAccNet. Set `multidomain` to `True` for the
  22 targets listed in Supplementary Table S7.
- The ROC table has `subset` (`all` or `multidomain`), `method`, `fpr`,
  `tpr`, `macro_auc`, and `n_targets`. It contains precomputed macro-averaged
  ROC curves built from residue-level predictions.
- The candidate set after excluding T1045s1 has 70 targets and 10,494 models.
  Per-method coverage reflects available predictions. The figure contains 17
  methods, and ProQ2 covers 68 targets in the all-target analysis.

## Figure 6 · CASP14 T1079 case study

```bash
python scripts/plot_casp14_fig6.py \
  --global-points evaluation/figure_3_3_target1079_global_points.csv \
  --true-lddt evaluation/T1079TS453_1.lddt \
  --deepaccnet evaluation/deepaccnet/T1079TS453_1.npz \
  --esm3 evaluation/esm3/T1079TS453_1.npy \
  --esm3-fusion evaluation/esm3_fusion/T1079TS453_1.npz \
  --structure-image evaluation/T1079_superposition.jpg \
  --output figures/Figure6_T1079.png
```

The global table contains 477 decoys and the columns `model`, `true`, `esm`,
`deep_esm`, and `deep`. The per-residue panels read the reference lDDT and
three prediction arrays for T1079TS453_1. The two NPZ archives use the `lddt`
key. The superposition image is supplied as a rendered input with the decoy in
pink and the native structure in blue. The script draws the legend and score
annotations over it.
