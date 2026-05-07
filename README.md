# SenseBench: A Benchmark for Remote Sensing Low-Level Visual Perception and Description in Large Vision-Language Models

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/Zhongchenchen/SenseBench)
> The dataset subset is available at [SenseBench_subset](https://huggingface.co/datasets/Zhongchenchen/SenseBench_subset) for portable access.

> We will regularly maintain and update SenseBench and this repository to foster a comprehensive remote sensing community.
---

## 📢 Latest Updates
- **`02-05-2026`**: We release the code for inference and evaluation.
- **`05-05-2026`**: We release the complete SenseBench benchmark in the [Hugging Face Dataset](https://huggingface.co/datasets/Zhongchenchen/SenseBench). 🔥🔥

---

## ✨ Overview

<p align="center">
  <img src="docs/imgs/perdes.png" width="100%" alt="Overview of the SenseBench evaluation framework">
</p>

<p align="justify">
  <b>Figure:</b> Overview of the SenseBench evaluation framework. The upper part shows the <i>SensePerception</i> task taxonomy across input formats, distortion settings, and <i>whether</i>/<i>what</i>/<i>how</i> question types. The lower part illustrates <i>SenseDescription</i> examples for single and paired inputs, where responses are evaluated by <i>completeness</i>, <i>correctness</i>, and <i>faithfulness</i>, with <span style="color:red;">red</span> text indicating incorrect or unsupported statements.
</p>

## 🏆 Observations

- **General-domain and RS-specialized VLMs exhibit divergent low-level perceptual priors, with no model achieving balanced high accuracy across both domains.** General-domain VLMs show stronger robustness to generic degradations but fail to generalize well to physics-driven remote sensing artifacts. Conversely, RSVLMs benefit from domain-specific tuning on RS-centric distortions, yet often sacrifice generic degradation robustness, revealing that model capacity alone does not resolve the domain gap.

- **Multi-distortion robustness remains a severe population-level bottleneck for modern VLMs.** Most models suffer substantial performance drops when multiple degradations are superimposed, including top single-distortion performers such as Gemini-3.1-pro-preview and EarthDial. This demonstrates that recognizing isolated artifacts does not imply the ability to disentangle compounded real-world degradations, making stratified multi-distortion evaluation essential.

- **Reference images act as structural anchors for paired perception, but their benefits are highly model-dependent.** General-domain VLMs generally leverage full-reference inputs to improve degradation detection, whereas RSVLMs often regress under paired-image settings, exposing weaknesses in multi-image spatial alignment. Meanwhile, strong single-image models such as InternVL3.5-38B can further improve with paired visual evidence, showing that single-image perception is not a reliable proxy for comparative reasoning.
<hr />

## 🗂️ Data Download

The SenseBench dataset is hosted on Hugging Face and can be downloaded locally for evaluation and reuse.

> Please refer to [docs/data.md](docs/data.md).

## 🛠️ Evaluation Workflow

The core inference and evaluation implementations live in `src/`.

> Please refer to [docs/evaluation.md](docs/evaluation.md).

<hr>

# 📜 Acknowledgements

We gratefully acknowledge the following data sources and platforms that made this project possible:

- **Google Earth** for high-resolution satellite imagery.  
- **Google Earth Engine (GEE)** for planetary-scale geospatial data access and processing tools.
- **Sentinel** satellite data, provided by the European Space Agency (ESA) via the Copernicus Open Access Hub.  
- **Landsat** satellite data, provided by the U.S. Geological Survey (USGS).
- **OpenStreetMap (OSM)** data, contributed by a global community of volunteers and accessed through various mapping services and APIs.

> Use of the Google Earth images must respect the [Google Earth terms of use](https://about.google/brand-resource-center/products-and-services/geo-guidelines/#google-earth). All images and their associated annotations in SenseBench can be used for academic purpose only, any commercial use is prohibited.

# 🏛 License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt the benchmark materials with appropriate attribution.
