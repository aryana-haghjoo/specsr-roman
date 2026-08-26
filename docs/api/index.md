# API reference

## Inference

```{eval-rst}
.. automodule:: specsr_roman.inference.pipeline
   :members: RomanPipeline, PipelineResult, build_sr2_input
```

## Models

```{eval-rst}
.. automodule:: specsr_roman.models.sr1
   :members:

.. automodule:: specsr_roman.models.zhead
   :members: ZHeadClf, ZHeadAttn, ZHead1D, make_z_grid, soft_labels, pz_stats, z_metrics, load_zhead

.. automodule:: specsr_roman.models.sr2
   :members: SR2Attention, topk_modes, constrain_delta, build_line_mask, line_profiles
```

## Data

```{eval-rst}
.. automodule:: specsr_roman.data.datasets
   :members:

.. automodule:: specsr_roman.data.splits
   :members:

.. automodule:: specsr_roman.data.transforms
   :members:

.. automodule:: specsr_roman.data.augment
   :members:

.. automodule:: specsr_roman.data.photometry
   :members:
```

## Training

```{eval-rst}
.. automodule:: specsr_roman.training.losses
   :members: loss_deblend_gated, line_flux_loss_weighted, line_hallucination_loss, line_flux_loss, presence_labels, sr2_reconstruction_loss, make_line_mask_from_smoothed

.. automodule:: specsr_roman.training.sr1
   :members: train

.. automodule:: specsr_roman.training.zhead
   :members: train

.. automodule:: specsr_roman.training.sr2
   :members: train
```

## Evaluation

```{eval-rst}
.. automodule:: specsr_roman.evaluation.cache
   :members:

.. automodule:: specsr_roman.evaluation.metrics
   :members:

.. automodule:: specsr_roman.evaluation.ablation
   :members: AblationConfig, run_ablation

.. automodule:: specsr_roman.evaluation.prior_dominance
   :members: PriorDominanceConfig, run_prior_dominance
```

## Extraction

```{eval-rst}
.. automodule:: specsr_roman.extraction.batch
   :members: ExtractionConfig, run_batch, run_worker, merge

.. automodule:: specsr_roman.extraction.frames
   :members:

.. automodule:: specsr_roman.extraction.catalog
   :members:

.. automodule:: specsr_roman.extraction.seds
   :members:

.. automodule:: specsr_roman.extraction.simulate
   :members:

.. automodule:: specsr_roman.extraction.extract
   :members:
```

## Configuration and constants

```{eval-rst}
.. automodule:: specsr_roman.config
   :members:

.. automodule:: specsr_roman.checkpoints
   :members:

.. automodule:: specsr_roman.grids
   :members: resolve_phot_tier

.. automodule:: specsr_roman.lines
   :members: count_strong_lines, angstrom_to_micron

.. automodule:: specsr_roman.paths
   :members:
```
