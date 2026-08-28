# specsr-roman

Physics-informed super-resolution of Roman grism spectra — recovering emission
lines without inventing them.

Roman's High Latitude Spectroscopic Survey will deliver slitless grism spectra
(R ≈ 461 λ/µm over 1–1.93 µm) for millions of emission-line galaxies. At that
resolving power the diagnostic complexes stay blended. `specsr-roman` super-resolves
them with a three-stage network, calibrated so that lines the data could not
have revealed are not drawn.

![The specsr-roman pipeline in three bands: simulating the training data (OpenUniverse2024 SEDs dispersed through the Wang+2022 grism model, extracted with grizli into 36,404 LR-HR pairs); spectral super-resolution (SR1, a conservative 1D residual CNN emitting mean and log-variance, refined by SR2, an attention network with one token per emission line); and the redshift branch (a ZHead taking the coarse spectrum plus three Roman bands and emitting P(z), whose top modes gate which lines SR2 may draw).](_static/arch_pipeline_roman.png)

```python
from specsr_roman import RomanPipeline

pipe = RomanPipeline.from_pretrained()
out = pipe.predict(flux_low, flux_low_err, phot=roman_medium_fluxes)
out.flux_sr, out.z, out.pz
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/installation
guides/quickstart
guides/data
guides/training
guides/evaluation
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/01_getting_started
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
changelog
```
