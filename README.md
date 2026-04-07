# OBS-Expert: Oracle Bone Script Interpretation with LLMs and Knowledge Graphs

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
</p>

<p align="center">
  <a href="README_CN.md"><img src="https://img.shields.io/badge/lang-中文-red.svg" alt="中文"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue.svg" alt="English"></a>
</p>

<p align="center">
  <em>A research framework for automated Oracle Bone Script (甲骨文) character analysis, combining vision models, Neo4j knowledge graphs, and multi-model LLM pipelines to generate scholarly interpretations of ancient Chinese characters.</em>
</p>

---

## Overview

Oracle Bone Script is the earliest known form of Chinese writing, inscribed on animal bones and turtle shells during the Shang dynasty (~1200 BCE). Interpreting these characters requires deep expertise in paleography. This project investigates whether LLMs, augmented with vision-based radical recognition and structured knowledge, can generate accurate character interpretations -- and how different augmentation strategies compare.

### Architecture

```
OB_Radix (source dataset: CSVs + images)
    |  sync_data.py
    v
experiment/data/ (working copy)
    |
    |---> PrototypeClassifier -- DinoV2 feature extraction -- radical prediction
    |---> KG_construct --------- Neo4j graph (radicals <-> characters)
    |---> chatgpt_rag ---------- RAG pipeline with tool-calling agents
    |---> cache_manager -------- LRU + disk caching for KG queries
    |
    v
experiment/exp*/run*.py (pipeline execution)
    |
    v
output/ (result CSVs)
```

## Dataset: OB_Radix

Due to size constraints, only a few example files are included under `OB_Radix/`. The full dataset contains:

| File | Description |
|------|-------------|
| `character_explanations_CN.csv` | Character -> scholarly interpretation (Chinese) |
| `character_explanations.csv` | Character -> scholarly interpretation |
| `character_analysis.csv` | Character -> type (pictographic / ideographic / pictophonetic) + reasoning |
| `radical_explanation.csv` | Radical -> definition + associated characters |

**Image directories:**

- `img_zi/{character}/` -- Oracle bone character images (`.jpg`) and extracted radical segments (`.png`)
- `organized_radicals/{radical}/` -- Radical exemplars for PrototypeClassifier training

## Experiments

### Exp 1: Radical Recognition

**Research question:** Can vision-based prototype matching classify oracle bone radicals?

Uses a **DinoV2** backbone to extract 768-dim feature vectors from radical images, then classifies via cosine similarity against learned prototypes.

- **Input:** Radical segment images
- **Output:** Top-k predicted radical classes with similarity scores

```bash
cd experiment/exp1
python run1.py
```

### Exp 2: Character Type Classification

**Research question:** Does knowledge graph augmentation improve character type classification?

Classifies characters into three types:

| Type | Chinese | Description |
|------|---------|-------------|
| Pictographic | 象形字 | Visually depicts the meaning |
| Ideographic compound | 会意字 | Combines semantic components |
| Pictophonetic | 形声字 | One component for meaning, one for sound |

Two pipelines are compared:

| Pipeline | Script | Description |
|----------|--------|-------------|
| Baseline | `run2_baseline.py` | Image -> LLM -> type prediction |
| Generation Module | `run2_generation_module.py` | Image -> PrototypeClassifier -> KG search -> LLM -> type prediction |

```bash
cd experiment/exp2
python run2_baseline.py
python run2_generation_module.py
```

### Exp 3: Interpretation Generation (Main Experiment)

**Research question:** Can LLMs generate accurate Oracle Bone Script interpretations? How do baseline, KG-augmented, and multi-agent approaches compare?

The dataset is split 70/30: the seen portion builds the knowledge graph, while the unseen portion tests generation quality.

| Pipeline | Script | Description |
|----------|--------|-------------|
| Baseline | `run_baseline.py` | Pure LLM generation (no external knowledge) |
| Prototype + KG | `run_prototype_kg.py` | DinoV2 radical prediction -> KG search -> augmented LLM prompt |
| Multi-Agent | `multi_agent_run.py` | Image Analysis Agent (tools + KG) -> Thinking Agent (deep reasoning) |

**Multi-Agent Architecture:**

```
Character Image + Radical Images
    |  [Image Analysis Agent + KG tools]
    v
Radical Predictions + KG Search Results
    |  [Thinking Agent with reasoning mode]
    v
5 Candidate Interpretations
```

```bash
cd experiment/exp3
python run_baseline.py
python run_prototype_kg.py
python multi_agent_run.py
```

Each pipeline produces 5 candidate interpretations per character.

### Supplementary Experiments

| Experiment | Script | Description |
|------------|--------|-------------|
| Variant Analysis | `analyze_variants.py` | Find character variants in both seen/unseen sets |
| Variant Processing | `var_run.py` | Oracle bone -> modern character mapping |
| English Version | `exp3_English_version/` | Exp 3 replicated with English-language prompts and outputs |

The English version includes pre-computed results for four models under `output/` (organized by `model_1/` through `model_4/`).

## Getting Started

### Prerequisites

- **Python** >= 3.10
- **Neo4j** database running on `localhost:7687`
- **GPU** recommended for DinoV2 inference (CPU fallback supported)

### Installation

```bash
git clone https://github.com/<your-org>/OBS_expert.git
cd OBS_expert
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Sync dataset to experiment directory
python tools/sync_data.py --src OB_Radix --targets experiment/data --with-assets

# 2. Set up your LLM API key
export LLM_API_KEY="your-api-key"
export LLM_MODEL="your-model-name"

# 3. Ensure Neo4j is running, then run any experiment
cd experiment/exp3
python run_prototype_kg.py
```

### LLM Configuration

All LLM calls are configurable via environment variables. Any LLM accessible via an OpenAI-compatible API can be used.

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL` | Model identifier | -- |
| `LLM_BASE_URL` | API endpoint (OpenAI-compatible) | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API key | -- |
| `LLM_TEMPERATURE` | Sampling temperature | `0.7` |
| `LLM_MAX_TOKENS` | Max output tokens | `4096` |
| `LLM_ENABLE_THINKING` | Enable reasoning mode (for models that support it) | `false` |
| `LLM_THINKING_MODELS` | Comma-separated list of model names that support thinking mode | -- |

## Project Structure

```
OBS_expert/
├── OB_Radix/                          # Source dataset
│   ├── img_zi/                        #   Character images (by character)
│   └── organized_radicals/            #   Radical exemplar images (by radical)
├── experiment/
│   ├── common/                        # Shared modules
│   │   ├── config.py                  #   Model loading & image handling
│   │   ├── chatgpt.py                 #   LLM vision+text API integration
│   │   ├── chatgpt_rag.py             #   RAG pipelines with tool-calling agents
│   │   ├── KG_construct.py            #   Neo4j knowledge graph construction
│   │   ├── PrototypeClassifier.py     #   DinoV2-based radical classification
│   │   ├── cache_manager.py           #   LRU + disk caching for KG queries
│   │   └── robust_csv_reader.py       #   Resilient CSV parsing
│   ├── exp1/                          # Exp 1: Radical Recognition
│   │   └── run1.py
│   ├── exp2/                          # Exp 2: Character Type Classification
│   │   ├── run2_baseline.py
│   │   └── run2_generation_module.py
│   ├── exp3/                          # Exp 3: Interpretation Generation
│   │   ├── run_baseline.py
│   │   ├── run_prototype_kg.py
│   │   └── multi_agent_run.py
│   ├── supplementary/                 # Supplementary experiments
│   │   ├── analyze_variants.py
│   │   ├── var_run.py
│   │   └── exp3_English_version/      #   English-language replication
│   └── data/                          # Synced working data (generated)
├── tools/
│   └── sync_data.py                   # OB_Radix -> experiment/data sync
├── requirements.txt
├── README.md                          # English
└── README_CN.md                       # 中文
```

## Version Control

- Track `OB_Radix/` and `experiment/common/` in version control
- `experiment/data/` is generated by `sync_data.py` -- ignored by default
- `output*/` directories in experiment folders are ignored by default

## Citation

If you find this work useful, please cite:

```bibtex
@article{obs_expert2025,
  title={OBS-Expert: Oracle Bone Script Interpretation with LLMs and Knowledge Graphs},
  author={},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
