# TM-score workflow

Reinstall the repository with `python -m pip install -e .` after updating it.
The TM-score commands use the same core environment as the lDDT predictors.

## Manifests

Each row describes one candidate model. Paths are resolved relative to the
manifest. Supply a reference `tm_score` in [0, 1] for every training and
validation row. For prediction, the reference column is optional. Labels
follow the reference definition of the input dataset. In the manuscript
CASP14 and CASP16 evaluation, the reference is the mean of the official
TM-scores over the assessed domains.

Standalone training and validation:

```csv
target,decoy,embedding,tm_score
target_1,model_1,embeddings/target_1/model_1.npz,0.81
```

DeepAccNet fusion training and validation:

```csv
target,decoy,feature,embedding,tm_score
target_1,model_1,deepaccnet/target_1/model_1.features.npz,embeddings/target_1/model_1.npz,0.81
```

For the DeepAccNet baseline, use `--framework fusion --feature deepaccnet`
with the fusion manifest columns `target,decoy,feature,tm_score`.
Inference uses the candidate-model features and the optional embedding.

Example manifests are available for the [standalone head](../examples/manifests/tm_head.csv)
and [fusion](../examples/manifests/tm_fusion.csv). To build manifests from
target/decoy trees, supply a CSV with `target,decoy,tm_score` columns:

```bash
python scripts/build_manifest.py --task tm \
  --embedding-root features/esm3 \
  --tm-labels labels/train_tm.csv --output manifests/tm_head_train.csv

python scripts/build_manifest.py --task tm \
  --embedding-root features/esm3 --feature-root features/deepaccnet \
  --tm-labels labels/train_tm.csv --output manifests/tm_fusion_train.csv
```

Omit `--tm-labels` to build an unlabeled inference manifest. For the
DeepAccNet baseline, provide only the DeepAccNet feature root.

## Training settings

- **Framework A:** residue-wise MLP with hidden widths 512, 256, and 128,
  followed by one scalar output per residue and mean pooling over residues.
  Defaults are 30 epochs, batch size 8, Adam at `1e-4`, and dropout 0.2. The
  validation MSE drives the ReduceLROnPlateau scheduler (factor 0.5, patience 2).
- **Framework B:** DeepAccNet pairwise trunk with five residual blocks and
  128 channels. The pretrained representation passes through a 512–256–64
  projection before fusion. Average the final pairwise features over both
  residue axes, then apply a 64-unit ELU layer and a scalar sigmoid output.
  Defaults are 200 epochs, batch size 1, and Adam at `5e-4` with 0.99
  learning-rate decay per epoch. The training and validation length limit is
  280 residues, exclusive. Inference evaluates the supplied manifest.
- The training loss is MSE, the default seed is 42, and each epoch samples one
  decoy per training target. Validation uses the lexicographically first
  decoy per target. `best.pt` is selected by validation MSE and `last.pt`
  stores the final epoch.

## Inference

The shipped model weights are in `weights/tm/head/` and `weights/tm/fusion/`.
Their metadata supplies the feature and framework automatically. Use
`weights/tm/fusion/deepaccnet.pt` with a feature-only manifest for the baseline.

For a historical fusion checkpoint without feature metadata, supply
`--feature`. Some older PyTorch
checkpoints also contain serialized NumPy training history. For checkpoints
whose source you trust, use `--allow-legacy-pickle` when the restricted loader
indicates it is required.

```bash
mqa-predict-tm --checkpoint path/to/best.pkl --feature esm3 \
  --manifest manifests/tm_fusion_test.csv --output outputs/legacy_tm_test \
  --allow-legacy-pickle
```

`metrics.json` reports pooled regression statistics when reference labels were
supplied. The manuscript benchmark summaries calculate correlations across
decoys within each target and average the resulting coefficients equally
across eligible targets.
