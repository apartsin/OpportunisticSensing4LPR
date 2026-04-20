# Mapping License Plate Recoverability Under Extreme Viewing Angles for Opportunistic Urban Sensing

Companion repository for the paper:

> Adamenko, I., Ben Aharon, O., Aperstein, Y., & Apartsin, A.
> *Mapping License Plate Recoverability Under Extreme Viewing Angles for Opportunistic Urban Sensing.*
> (2026, preprint.)

The paper introduces **recoverability maps**, a task-agnostic framework for quantifying where in a parameterised degradation space an opportunistic-sensing task remains reliable. The framework is demonstrated on oblique-view license plate recognition (LPR) as a concrete first instance, comparing five deep-learning restoration architectures (U-Net, angle-conditioned U-Net, Restormer, Pix2Pix, SR3 diffusion) on a shared full-angle grid.

## Read the paper

- **HTML (web-rendered, with interactive math):** [`index.html`](index.html)
- **MS Word (.docx, native OMML equations, editable):** [`Adamenko_et_al_Opportunistic_LPR_draft1.docx`](Adamenko_et_al_Opportunistic_LPR_draft1.docx)

The HTML version renders fully in any modern browser via KaTeX. A published GitHub Pages site serves it directly: see [`https://apartsin.github.io/OpportunisticSensing4LPR`](https://apartsin.github.io/OpportunisticSensing4LPR) once Pages is enabled for this repository.

## What is in this repository

```
.
├── index.html                                   paper (HTML source, KaTeX-rendered)
├── Adamenko_et_al_Opportunistic_LPR_draft1.docx paper (MS Word, native equations)
├── figures/
│   ├── city_opportunistic_sensing.png           Fig. 1 (concept illustration)
│   └── generated/
│       ├── fig01_psnr_ssim_combined.png         Fig. 4 (PSNR/SSIM per model)
│       ├── fig02_dataset_angle_distribution.png Fig. 2 (training PDF surfaces)
│       ├── fig04_auc_f_slopegraph.png           Fig. 5 (dataset-shift sensitivity)
│       ├── fig07_psnr_ocr_correlation.png       Fig. 6 (PSNR-OCR scatter)
│       ├── fig08_ssim_ocr_threshold.png         Fig. 7 (SSIM-OCR threshold)
│       ├── fig_comparison_panel.png             Fig. 8 (qualitative reconstructions)
│       └── fig_synth_pipeline_panel.png         Fig. 3 (data-construction pipeline)
├── scripts/
│   ├── generate_figures.py                      reproduces Figs. 2, 4, 5
│   ├── make_pipeline_panel.py                   reproduces Fig. 3
│   ├── make_comparison_panel.py                 reproduces Fig. 8
│   └── make_ocr_correlation_figures.py          reproduces Figs. 6, 7 from report data
├── models/
│   ├── unet.py                                  U-Net baseline architecture
│   ├── unet_conditional.py                      Angle-conditioned U-Net (FiLM)
│   ├── restormer.py                             Restormer (MDTA transformer)
│   ├── pix2pix.py                               Pix2Pix GAN generator + discriminator
│   └── diffusion_sr3.py                         SR3 diffusion (velocity prediction, DDIM)
└── notebooks/
    ├── Sampling.ipynb                           Scrambled Sobol angle sampling,
    │                                            PDF parameterisation, dataset construction
    ├── ocr_test.ipynb                           Tesseract OCR harness (preprocessing,
    │                                            per-digit fallbacks, plate-level scoring)
    ├── results.ipynb                            Per-angle-pair evaluation, AUC / F
    │                                            computation, architecture comparison
    └── report_artifacts.ipynb                   Original data-extraction notebook used
                                                 to derive the tables reported in the paper
```

## Reproducing the figures

All figures in the paper can be re-produced from the scripts in `scripts/`. Requirements: Python ≥ 3.10 with `numpy`, `matplotlib`, and `Pillow`.

```bash
cd scripts
python generate_figures.py
python make_pipeline_panel.py
python make_comparison_panel.py
python make_ocr_correlation_figures.py
```

## Dataset construction and training

- **`notebooks/Sampling.ipynb`** documents the Scrambled Sobol angle-sampling procedure, the parametric PDFs used for the Standard (DS-S) and Extreme (DS-E) datasets, and the deterministic pipeline that rebuilds the 80/10/10 splits bit-identically from a seed.
- **`models/`** contains the five restoration architectures compared in the paper. Each file is a standalone model definition that follows the training protocol of Section 4.3 (AdamW / Adam with cosine or warm-up schedule, batch sizes as reported in Table 2).
- **`notebooks/results.ipynb`** runs the full-grid evaluation described in Section 4.4, computes per-angle-pair plate-level OCR accuracy, PSNR, and SSIM, and produces the boundary-AUC and reliability-$F$ summary values reported in Table 4.
- **`notebooks/ocr_test.ipynb`** encapsulates the Tesseract v4 OCR harness (grayscale, contrast normalisation, $2\times$ upscaling, binarisation, per-digit Otsu / adaptive / colour-inversion fallbacks) described in Section 4.4.

The synthetic training datasets themselves are **not** redistributed as static files. They are fully deterministic and bit-identical given the Sobol seeds and distortion-pipeline parameters documented in Section 4.1 of the paper and in `notebooks/Sampling.ipynb`; re-running the sampling notebook regenerates them exactly.

A DOI-minted Zenodo snapshot of this repository will accompany the final acceptance.

## Citation

Please cite the paper as:

```bibtex
@article{Adamenko2026recoverability,
  title  = {Mapping License Plate Recoverability Under Extreme Viewing Angles for Opportunistic Urban Sensing},
  author = {Adamenko, Igor and Ben Aharon, Orpaz and Aperstein, Yehudit and Apartsin, Alexander},
  year   = {2026},
  note   = {Preprint}
}
```

## Contact

Corresponding author: Alexander Apartsin
School of Computer Science, Faculty of Sciences, HIT – Holon Institute of Technology, Holon 58102, Israel.

## License

The paper text and figures are made available under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. The scripts in `scripts/` are released under the [MIT License](https://opensource.org/licenses/MIT).
