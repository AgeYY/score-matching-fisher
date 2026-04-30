# PR-autoencoder high-dimensional datasets

## Breaking change

The dataset family token `randamp_gaussian_sqrtd_pr_autoencoder` is **removed** from `bin/make_dataset.py`, `fisher/cli_shared_fisher.py`, and NPZ recipes. Old archives that still declare this `dataset_family` in JSON meta will raise a clear error on load.

## Two-step workflow

1. **Low-dimensional generative data** (tuning-curve figures, native `x_dim`):

   ```bash
   python bin/make_dataset.py --dataset-family randamp_gaussian_sqrtd --x-dim 2 ... --output-npz data/lowdim.npz
   ```

2. **Embed into high-dimensional observation space** (one four-panel figure next to the output NPZ: `pr_projection_summary.{png,svg}` — same layout as `make_dataset.py` tuning + manifold + scatter for embedded $x$, plus PR-autoencoder loss vs epoch):

   ```bash
   python bin/project_dataset_pr_autoencoder.py \
     --input-npz data/lowdim.npz \
     --output-npz data/highdim.npz \
     --h-dim 10 \
     --device cuda
   ```

By default the projection script **always trains** a PR-autoencoder (no checkpoint reuse). Pass **`--use-cache`** to load a matching cached model from `--cache-dir` when available.

The embedded NPZ keeps `dataset_family: randamp_gaussian_sqrtd`, sets `meta["x_dim"]` to `--h-dim`, and sets `pr_autoencoder_embedded: true` with `pr_autoencoder_z_dim` equal to the source latent dimension. `build_dataset_from_meta` uses the latent dimension for the generative toy class when `pr_autoencoder_embedded` is true.
