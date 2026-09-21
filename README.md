# mqa

`mqa` predicts residue-level protein model quality from pretrained protein
representations, either with a small regression head or by fusing the
representation into DeepAccNet.

The project contains the models used in the experiments for ESM-2, ESM-3,
SaProt, ProstT5, ESM-IF1, and ProteinMPNN.

## Models

| Feature | Checkpoint | Residue tensor | Standalone MLP | DeepAccNet projection |
| --- | --- | ---: | --- | --- |
| ESM-2 | `esm2_t33_650M_UR50D` | `L x 1280` | 512-256-128 | 512-256-64 |
| ESM-3 | `esm3-open` | `L x 1536` | 512-256-128 | 512-256-64 |
| SaProt | `SaProt_650M_AF2` | `L x 1280` | 512-256-128 | 512-256-64 |
| ProstT5 | `Rostlab/ProstT5` | `L x 1024` | 512-256-128 | 512-256-64 |
| ESM-IF1 | `esm_if1_gvp4_t16_142M_UR50` | `L x 512` | 256-128 | 256-64 |
| ProteinMPNN | `v_48_020` | `L x 128` | 256-128 | 64 |

The default standalone model is the MLP that was used in the experiments. A
true single-layer baseline is available with `--head linear`. ProteinMPNN is a
structure-conditioned sequence-design model rather than a protein language
model, but its encoder representation is handled through the same interface.

The merged DeepAccNet implementation is checkpoint-compatible with all six
original experiment models.

## Installation

Create the training environment:

```bash
conda env create -f environments/core.yml
conda activate mqa-core
```

Feature extraction uses separate reproducible environments because fair-esm and
ESM-3 both provide an `esm` Python package:

```bash
conda env create -f environments/esm2_esmif1.yml
conda env create -f environments/esm3.yml
conda env create -f environments/saprot.yml
conda env create -f environments/prostt5.yml
conda env create -f environments/proteinmpnn.yml
conda env create -f environments/deepaccnet.yml
```

See [environment notes](environments/README.md) for CUDA, Foldseek, PyRosetta,
and weight boundaries.

SaProt additionally requires the Foldseek executable. ProteinMPNN requires an
upstream [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) checkout and its
`v_48_020.pt` checkpoint. DeepAccNet feature generation requires an upstream
[DeepAccNet](https://github.com/hiranumn/DeepAccNet) checkout and a separately
licensed PyRosetta installation.

## Feature extraction

All extractors write compressed NPZ files with one required array:

```text
embedding: float32 [L, D]
```

Metadata such as `model_name`, `source`, `sequence`, and `chain_ids` is stored
when available.

Sequence models take FASTA input:

```bash
mqa-extract --feature esm2 --input proteins.fasta --output features/esm2
mqa-extract --feature prostt5 --input proteins.fasta --output features/prostt5
```

Structure models take one PDB/mmCIF file or a directory tree:

```bash
mqa-extract --feature esm3 --input structures --output features/esm3
mqa-extract --feature esmif1 --input structures --output features/esmif1
mqa-extract --feature saprot --input structures --output features/saprot \
  --model-path /path/to/SaProt_650M_AF2 --foldseek /path/to/foldseek
mqa-extract --feature proteinmpnn --input structures --output features/proteinmpnn \
  --proteinmpnn-repo /path/to/ProteinMPNN
```

ESM-3 processes all chains by default; pass `--chain A` to restrict it to one
chain. The other structure extractors also accept `--chain`.

The ProteinMPNN extractor captures the last encoder layer through a forward
hook, so the upstream repository does not need to be patched.

## Manifests

Training and testing are driven by CSV manifests, which keeps machine-specific
paths out of the source. Relative paths are resolved from the manifest file.

Standalone head manifest:

```csv
target,decoy,embedding,label
protein_1,model_1,features/protein_1/model_1.npz,labels/protein_1/model_1.json
```

Fusion manifest:

```csv
target,decoy,feature,embedding,native
protein_1,model_1,deepaccnet/protein_1/model_1.features.npz,features/protein_1/model_1.npz,deepaccnet/protein_1/native.features.npz
```

Labels may be JSON files with `local_lddt[chain][residue_index]` values, or NPZ
files with a `local_lddt`/`lddt` array and optional boolean `mask`. A helper can
build manifests from matching directory trees:

```bash
python scripts/build_manifest.py \
  --embedding-root features/esm3 \
  --label-root labels \
  --output manifests/esm3_train.csv
```

## Standalone head

```bash
mqa-train-head \
  --feature esm3 \
  --train-manifest manifests/train.csv \
  --valid-manifest manifests/valid.csv \
  --output outputs/esm3_head

mqa-test-head \
  --checkpoint outputs/esm3_head/best.pt \
  --manifest manifests/test.csv \
  --output outputs/esm3_head_test
```

Use `--head linear` for the additional `Linear(D, 1)` baseline.
For an old state-dict-only checkpoint, add its representation name, for example
`--feature esm3`. The existing ESM-3, SaProt, ProstT5, ESM-IF1, and ProteinMPNN
head weights are supported without conversion. The ESM-2 standalone entrypoint
is newly unified from the same 1280-dimensional head layout; no historical ESM-2
standalone result is claimed.

## DeepAccNet fusion

DeepAccNet features must first be generated as `.features.npz` files with the
original DeepAccNet program. MQA reads these files but does not generate them.

```bash
mqa-train-fusion \
  --feature saprot \
  --train-manifest manifests/fusion_train.csv \
  --valid-manifest manifests/fusion_valid.csv \
  --output outputs/saprot_fusion

mqa-predict-fusion \
  --checkpoint outputs/saprot_fusion/best.pkl \
  --manifest manifests/fusion_test.csv \
  --output outputs/saprot_fusion_test
```

Legacy experiment checkpoints do not contain feature metadata. Supply the
matching feature explicitly when predicting:

```bash
mqa-predict-fusion --checkpoint /path/to/best.pkl --feature esm3 \
  --manifest manifests/fusion_test.csv --output outputs/esm3_fusion_test \
  --allow-legacy-pickle
```

The final flag is required only for trusted historical DeepAccNet checkpoints
that contain NumPy training-history objects and cannot use PyTorch's restricted
weights-only loader.

## Weights and data

Model weights, datasets, structure files, extracted tensors, and prediction
outputs are not included. Keep them in external directories and pass their
paths through command-line arguments or manifests.

## References

- [DeepAccNet](https://github.com/hiranumn/DeepAccNet)
- [ESM-2 and ESM-IF1](https://github.com/facebookresearch/esm)
- [ESM-3](https://github.com/evolutionaryscale/esm)
- [SaProt](https://github.com/westlake-repl/SaProt)
- [ProstT5](https://huggingface.co/Rostlab/ProstT5)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
