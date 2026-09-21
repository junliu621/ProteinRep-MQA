# Reproducible environments

Use separate environments for extraction backends. In particular, fair-esm and
ESM-3 both install an import package named `esm` and must not share an
environment.

Create an environment from the repository root:

```bash
conda env create -f environments/core.yml
conda env create -f environments/esm2_esmif1.yml
conda env create -f environments/esm3.yml
conda env create -f environments/saprot.yml
conda env create -f environments/prostt5.yml
conda env create -f environments/proteinmpnn.yml
conda env create -f environments/deepaccnet.yml
```

The pinned application/library versions follow the environments in which the
project code was developed. PyTorch is pinned to a stable release in the public
files instead of the machine-specific CUDA nightly builds that were present in
some local environments. Select a PyTorch/CUDA build compatible with the host
driver when GPU extraction is required.

Observed local environments are recorded below for provenance. The `+cu128`
and development builds are hardware-specific observations, not portable pins:

| Task | Python | PyTorch | NumPy | Main packages |
| --- | --- | --- | --- | --- |
| Head training | 3.10.19 | 2.6.0+cu124 | 2.2.6 | SciPy 1.15.3, tqdm 4.67.3 |
| ESM-2 | 3.10.20 | 2.12.0.dev+cu128 | 2.2.6 | fair-esm 2.0.0 |
| ESM-IF1 | 3.9.25 | 2.8.0+cu128 | 1.26.4 | fair-esm 2.0.0 |
| SaProt | 3.10.20 | 2.12.0.dev+cu128 | 1.25.2 | transformers 4.28.0, tokenizers 0.13.3, Biopython 1.81 |
| ProstT5 | 3.10.20 | 2.7.1+cu128 | 1.26.4 | transformers 4.38.2, tokenizers 0.15.2, sentencepiece 0.2.1 |
| ProteinMPNN | 3.10.20 | 2.12.0.dev+cu128 | 2.2.6 | Biopython 1.86 |
| DeepAccNet | 3.10.20 | 2.11.0+cu128 | 2.2.6 | SciPy 1.15.3, PyRosetta 2024.18 |

PyRosetta is deliberately absent from `deepaccnet.yml`. It has separate license
terms and must be installed from an authorized PyRosetta distribution. Foldseek
is an external executable and must be installed separately for SaProt.

Model weights are not environment dependencies and are never downloaded by the
environment files.

Source checkouts used by adapters can be pinned independently:

```bash
git clone https://github.com/dauparas/ProteinMPNN.git
git -C ProteinMPNN checkout 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57

git clone https://github.com/hiranumn/DeepAccNet.git
git -C DeepAccNet checkout cc3c191d03b35352f6b49bf6162b275f815c8244
```

The ESM-3 environment installs upstream commit
`6e89bf7d127b06e95d8243c020936cd67d862229`. The inspected SaProt utility
checkout was `e91e4858b55944523f1f8d385f7b96a0d3d34c1d`; the extractor itself uses
the Hugging Face checkpoint plus an external Foldseek executable.
