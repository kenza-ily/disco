#!/usr/bin/env python3
"""
Script to create parsing_analysis.ipynb and qa_analysis.ipynb
These notebooks aggregate results across datasets and answer 3 key questions:
1. Which is the best phase for each dataset?
2. Which is the best model per phase for each dataset?
3. What's the best combination (phase + model) for each dataset?
"""

import json
from pathlib import Path

def create_notebook_cell(cell_type, content, metadata=None):
    """Create a notebook cell"""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": content if isinstance(content, list) else [content]
    }

    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

    return cell

def create_parsing_notebook():
    """Create parsing_analysis.ipynb"""

    cells = []

    # Cell 1: Executive Summary
    cells.append(create_notebook_cell("markdown", [
        "# Parsing Tasks Aggregated Analysis\n",
        "\n",
        "## Executive Summary\n",
        "\n",
        "**Objective:** Aggregate and compare parsing performance across all datasets to answer 3 key questions:\n",
        "\n",
        "### 🎯 Key Questions:\n",
        "1. **Which is the best phase for each dataset?**\n",
        "2. **Which is the best model per phase for each dataset?**\n",
        "3. **What's the best combination (phase + model) for each dataset?**\n",
        "\n",
        "**Datasets Included:**\n",
        "- ✅ **IAM_mini** (500 samples, 3 phases) - Handwriting Recognition\n",
        "- ✅ **ICDAR_mini** (491 samples, 3 phases) - Multilingual OCR\n",
        "- ✅ **VOC2007** (238 samples, 4 phases) - Chinese Medical Text\n",
        "- ✅ **RX-PAD** (200 samples, 3 phases) - French Medical Forms\n",
        "\n",
        "**Evaluation Metrics:**\n",
        "- 🎯 **PRIMARY: Cosine Similarity** (semantic similarity, higher is better)\n",
        "- **SECONDARY:** CER (Character Error Rate, lower is better), WER (Word Error Rate, lower is better)\n",
        "\n",
        "**Phases:**\n",
        "- **P-A (Pa):** OCR Baseline\n",
        "- **P-B (Pb):** VLM with generic prompts\n",
        "- **P-C (Pc):** VLM with task-aware prompts\n",
        "- **P-D (Pd):** VLM with detailed context (VOC2007 only)\n"
    ]))

    # Cell 2: Imports
    cells.append(create_notebook_cell("code", [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import json\n",
        "import sys\n",
        "from pathlib import Path\n",
        "from typing import List, Dict, Optional\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "from tqdm.notebook import tqdm\n",
        "\n",
        "# Setup paths\n",
        "NOTEBOOK_DIR = Path.cwd()\n",
        "REPO_ROOT = NOTEBOOK_DIR.parent.parent.parent\n",
        "\n",
        "if str(REPO_ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(REPO_ROOT))\n",
        "\n",
        "# Import metrics\n",
        "from ocr_vs_vlm.metrics.evaluation_metrics import calculate_cer, calculate_wer\n",
        "from ocr_vs_vlm.metrics.embedding_cache import EmbeddingCacheManager, save_embeddings_for_phase\n",
        "\n",
        "# Plotting style\n",
        "plt.style.use('seaborn-v0_8-whitegrid')\n",
        "plt.rcParams['figure.figsize'] = (14, 6)\n",
        "plt.rcParams['font.size'] = 11\n",
        "sns.set_palette('husl')\n",
        "\n",
        "# Paths\n",
        "RESULTS_BASE = REPO_ROOT / 'ocr_vs_vlm' / 'results'\n",
        "EMBEDDINGS_DIR = RESULTS_BASE / '3_embeddings'\n",
        "\n",
        "print(\"✅ Libraries loaded successfully!\")\n",
        "print(f\"📂 Results base: {RESULTS_BASE}\")\n",
        "print(f\"📂 Embeddings: {EMBEDDINGS_DIR}\")\n"
    ]))

    # Cell 3: Dataset Configuration
    cells.append(create_notebook_cell("code", [
        "# Dataset configuration\n",
        "PARSING_DATASETS = {\n",
        "    'IAM_mini': {\n",
        "        'path': RESULTS_BASE / '2_clean' / 'IAM_mini',\n",
        "        'task': 'handwriting',\n",
        "        'phases': ['Pa', 'Pb', 'Pc'],\n",
        "        'chunk_size': None\n",
        "    },\n",
        "    'ICDAR_mini': {\n",
        "        'path': RESULTS_BASE / '2_clean' / 'ICDAR_mini',\n",
        "        'task': 'multilingual',\n",
        "        'phases': ['Pa', 'Pb', 'Pc'],\n",
        "        'chunk_size': 150\n",
        "    },\n",
        "    'VOC2007': {\n",
        "        'path': RESULTS_BASE / '2_clean' / 'VOC2007',\n",
        "        'task': 'medical_chinese',\n",
        "        'phases': ['Pa', 'Pb', 'Pc', 'Pd'],\n",
        "        'chunk_size': 200\n",
        "    },\n",
        "    'RX-PAD': {\n",
        "        'path': RESULTS_BASE / '2_clean' / 'RX-PAD',\n",
        "        'task': 'medical_french',\n",
        "        'phases': ['Pa', 'Pb', 'Pc'],\n",
        "        'chunk_size': 200\n",
        "    }\n",
        "}\n",
        "\n",
        "print(\"📁 Dataset Configuration:\")\n",
        "for name, config in PARSING_DATASETS.items():\n",
        "    print(f\"  - {name}: {len(config['phases'])} phases, task={config['task']}\")\n"
    ]))

    # Cell 4: Helper Functions
    cells.append(create_notebook_cell("code", [
        "def is_valid_row(row, pred_col: str, err_col: Optional[str] = None) -> bool:\n",
        "    \"\"\"Check if prediction is valid (non-empty, no error)\"\"\"\n",
        "    pred_value = row[pred_col]\n",
        "    if pd.isna(pred_value) or str(pred_value).strip() == \"\":\n",
        "        return False\n",
        "    if err_col and err_col in row.index:\n",
        "        if pd.notna(row[err_col]) and str(row[err_col]).strip() != \"\":\n",
        "            return False\n",
        "    return True\n",
        "\n",
        "def calculate_metrics(gt: str, pred: str, phase: str, sample_id: str, \n",
        "                     model: str, emb_manager: EmbeddingCacheManager) -> Dict:\n",
        "    \"\"\"Calculate CER, WER, and Cosine Similarity\"\"\"\n",
        "    if pd.isna(pred) or str(pred).strip() == \"\":\n",
        "        return {'cer': 1.0, 'wer': 1.0, 'cosine_similarity': 0.0}\n",
        "    \n",
        "    gt_str = str(gt)\n",
        "    pred_str = str(pred)\n",
        "    \n",
        "    return {\n",
        "        'cer': float(calculate_cer(gt_str, pred_str)),\n",
        "        'wer': float(calculate_wer(gt_str, pred_str)),\n",
        "        'cosine_similarity': float(emb_manager.compute_cosine_similarity(\n",
        "            phase=phase, ground_truth=gt_str, prediction=pred_str,\n",
        "            sample_id=sample_id, model=model\n",
        "        ))\n",
        "    }\n",
        "\n",
        "print(\"✅ Helper functions defined\")\n"
    ]))

    # Cell 5: Load Data
    cells.append(create_notebook_cell("code", [
        "# Load all datasets\n",
        "all_data = {}\n",
        "embedding_managers = {}\n",
        "\n",
        "for dataset_name, config in PARSING_DATASETS.items():\n",
        "    print(f\"\\nLoading {dataset_name}...\")\n",
        "    \n",
        "    # Initialize embedding manager\n",
        "    emb_manager = EmbeddingCacheManager(dataset_name, EMBEDDINGS_DIR)\n",
        "    embedding_managers[dataset_name] = emb_manager\n",
        "    \n",
        "    dataset_dfs = {}\n",
        "    for phase in config['phases']:\n",
        "        file_path = config['path'] / f\"{phase}.csv\"\n",
        "        if file_path.exists():\n",
        "            df = pd.read_csv(file_path)\n",
        "            dataset_dfs[phase] = df\n",
        "            print(f\"  ✅ {phase}: {len(df)} samples\")\n",
        "    \n",
        "    all_data[dataset_name] = {\n",
        "        'config': config,\n",
        "        'phase_dfs': dataset_dfs\n",
        "    }\n",
        "\n",
        "print(f\"\\n✅ Loaded {len(all_data)} datasets\")\n"
    ]))

    # Cell 6: Calculate Metrics
    cells.append(create_notebook_cell("code", [
        "# Calculate metrics for all datasets\n",
        "all_metrics = []\n",
        "\n",
        "for dataset_name, data in all_data.items():\n",
        "    print(f\"\\nCalculating metrics for {dataset_name}...\")\n",
        "    config = data['config']\n",
        "    emb_manager = embedding_managers[dataset_name]\n",
        "    \n",
        "    for phase, df in data['phase_dfs'].items():\n",
        "        pred_cols = [col for col in df.columns if col.startswith('prediction_')]\n",
        "        \n",
        "        for pred_col in pred_cols:\n",
        "            model = pred_col.replace('prediction_', '')\n",
        "            err_col = f'error_{model}'\n",
        "            \n",
        "            # Get valid rows\n",
        "            valid_rows = [r for _, r in df.iterrows() if is_valid_row(r, pred_col, err_col)]\n",
        "            \n",
        "            if not valid_rows:\n",
        "                continue\n",
        "            \n",
        "            # Calculate metrics\n",
        "            metrics_list = []\n",
        "            for row in tqdm(valid_rows, desc=f\"  {phase}/{model}\", leave=False):\n",
        "                m = calculate_metrics(\n",
        "                    row['ground_truth'], row[pred_col], phase,\n",
        "                    row['sample_id'], model, emb_manager\n",
        "                )\n",
        "                metrics_list.append(m)\n",
        "            \n",
        "            # Aggregate\n",
        "            all_metrics.append({\n",
        "                'dataset': dataset_name,\n",
        "                'task': config['task'],\n",
        "                'phase': phase,\n",
        "                'model': model,\n",
        "                'cosine_similarity': np.mean([m['cosine_similarity'] for m in metrics_list]),\n",
        "                'cer': np.mean([m['cer'] for m in metrics_list]),\n",
        "                'wer': np.mean([m['wer'] for m in metrics_list]),\n",
        "                'valid_samples': len(valid_rows)\n",
        "            })\n",
        "\n",
        "metrics_df = pd.DataFrame(all_metrics)\n",
        "print(f\"\\n✅ Calculated metrics for {len(metrics_df)} combinations\")\n"
    ]))

    # Cell 7: SECTION 3 - Best Phase per Dataset (KEY QUESTION #1)
    cells.append(create_notebook_cell("markdown", [
        "## 🎯 Section 3: Best Phase per Dataset\n",
        "\n",
        "**KEY QUESTION #1: Which is the best phase for each dataset?**\n",
        "\n",
        "This section identifies the best-performing phase for each dataset based on the PRIMARY metric (Cosine Similarity).\n"
    ]))

    cells.append(create_notebook_cell("code", [
        "# Identify best phase per dataset\n",
        "print(\"=\"*80)\n",
        "print(\"BEST PHASE PER DATASET (by Cosine Similarity - PRIMARY METRIC)\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "best_phases = []\n",
        "\n",
        "for dataset in metrics_df['dataset'].unique():\n",
        "    dataset_data = metrics_df[metrics_df['dataset'] == dataset]\n",
        "    \n",
        "    # Average across models per phase\n",
        "    phase_avg = dataset_data.groupby('phase').agg({\n",
        "        'cosine_similarity': 'mean',\n",
        "        'cer': 'mean',\n",
        "        'wer': 'mean'\n",
        "    }).round(4)\n",
        "    \n",
        "    best_phase = phase_avg['cosine_similarity'].idxmax()\n",
        "    best_metrics = phase_avg.loc[best_phase]\n",
        "    \n",
        "    best_phases.append({\n",
        "        'Dataset': dataset,\n",
        "        'Best Phase': best_phase,\n",
        "        'Cosine Similarity': best_metrics['cosine_similarity'],\n",
        "        'CER': best_metrics['cer'],\n",
        "        'WER': best_metrics['wer']\n",
        "    })\n",
        "    \n",
        "    print(f\"\\n{dataset}:\")\n",
        "    print(f\"  🏆 Best Phase: {best_phase}\")\n",
        "    print(f\"  🎯 Cosine Similarity: {best_metrics['cosine_similarity']:.4f}\")\n",
        "    print(f\"     CER: {best_metrics['cer']:.4f}, WER: {best_metrics['wer']:.4f}\")\n",
        "\n",
        "best_phases_df = pd.DataFrame(best_phases)\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"SUMMARY TABLE:\")\n",
        "display(best_phases_df)\n"
    ]))

    # Cell 8: Visualize Best Phases
    cells.append(create_notebook_cell("code", [
        "# Visualize best phase per dataset\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "\n",
        "# Bar chart\n",
        "x = range(len(best_phases_df))\n",
        "bars = ax.bar(x, best_phases_df['Cosine Similarity'], color='steelblue', alpha=0.8)\n",
        "\n",
        "# Highlight bars\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels([f\"{row['Dataset']}\\n({row['Best Phase']})\" \n",
        "                     for _, row in best_phases_df.iterrows()])\n",
        "ax.set_ylabel('Cosine Similarity (PRIMARY METRIC)', fontweight='bold')\n",
        "ax.set_title('🏆 Best Phase per Dataset', fontsize=14, fontweight='bold')\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "\n",
        "# Add value labels\n",
        "for i, bar in enumerate(bars):\n",
        "    height = bar.get_height()\n",
        "    ax.text(bar.get_x() + bar.get_width()/2., height,\n",
        "            f'{height:.4f}', ha='center', va='bottom')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    # Cell 9: SECTION 4 - Best Model per Phase per Dataset (KEY QUESTION #2)
    cells.append(create_notebook_cell("markdown", [
        "## 🎯 Section 4: Best Model per Phase per Dataset\n",
        "\n",
        "**KEY QUESTION #2: Which is the best model per phase for each dataset?**\n",
        "\n",
        "This section identifies the best-performing model for each dataset-phase combination.\n"
    ]))

    cells.append(create_notebook_cell("code", [
        "# Identify best model per phase per dataset\n",
        "print(\"=\"*80)\n",
        "print(\"BEST MODEL PER PHASE PER DATASET\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "best_models_per_phase = []\n",
        "\n",
        "for dataset in metrics_df['dataset'].unique():\n",
        "    print(f\"\\n{dataset}:\")\n",
        "    dataset_data = metrics_df[metrics_df['dataset'] == dataset]\n",
        "    \n",
        "    for phase in dataset_data['phase'].unique():\n",
        "        phase_data = dataset_data[dataset_data['phase'] == phase]\n",
        "        best_idx = phase_data['cosine_similarity'].idxmax()\n",
        "        best = phase_data.loc[best_idx]\n",
        "        \n",
        "        best_models_per_phase.append({\n",
        "            'Dataset': dataset,\n",
        "            'Phase': phase,\n",
        "            'Best Model': best['model'],\n",
        "            'Cosine Similarity': best['cosine_similarity'],\n",
        "            'CER': best['cer'],\n",
        "            'WER': best['wer']\n",
        "        })\n",
        "        \n",
        "        print(f\"  {phase}: {best['model']} (Cosine={best['cosine_similarity']:.4f})\")\n",
        "\n",
        "best_models_df = pd.DataFrame(best_models_per_phase)\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"SUMMARY TABLE:\")\n",
        "display(best_models_df)\n"
    ]))

    # Cell 10: SECTION 5 - Best Combination per Dataset (KEY QUESTION #3)
    cells.append(create_notebook_cell("markdown", [
        "## 🎯 Section 5: Best Combination per Dataset\n",
        "\n",
        "**KEY QUESTION #3: What's the best combination (phase + model) for each dataset?**\n",
        "\n",
        "This section identifies the single best phase-model combination for each dataset.\n"
    ]))

    cells.append(create_notebook_cell("code", [
        "# Find best combination per dataset\n",
        "print(\"=\"*80)\n",
        "print(\"BEST COMBINATION (Phase + Model) PER DATASET\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "best_combinations = []\n",
        "\n",
        "for dataset in metrics_df['dataset'].unique():\n",
        "    dataset_data = metrics_df[metrics_df['dataset'] == dataset]\n",
        "    best_idx = dataset_data['cosine_similarity'].idxmax()\n",
        "    best = dataset_data.loc[best_idx]\n",
        "    \n",
        "    # Get baseline (Pa phase) for comparison\n",
        "    baseline_data = dataset_data[dataset_data['phase'] == 'Pa']\n",
        "    if len(baseline_data) > 0:\n",
        "        baseline_cosine = baseline_data['cosine_similarity'].mean()\n",
        "        improvement = ((best['cosine_similarity'] - baseline_cosine) / baseline_cosine * 100) if baseline_cosine > 0 else 0\n",
        "    else:\n",
        "        improvement = 0\n",
        "    \n",
        "    best_combinations.append({\n",
        "        'Dataset': dataset,\n",
        "        'Best Phase': best['phase'],\n",
        "        'Best Model': best['model'],\n",
        "        'Cosine Similarity': best['cosine_similarity'],\n",
        "        'CER': best['cer'],\n",
        "        'WER': best['wer'],\n",
        "        'Improvement vs Baseline (%)': round(improvement, 2)\n",
        "    })\n",
        "    \n",
        "    print(f\"\\n{dataset}:\")\n",
        "    print(f\"  🏆 Best Combination: {best['phase']} + {best['model']}\")\n",
        "    print(f\"  🎯 Cosine Similarity: {best['cosine_similarity']:.4f}\")\n",
        "    print(f\"     CER: {best['cer']:.4f}, WER: {best['wer']:.4f}\")\n",
        "    if improvement != 0:\n",
        "        print(f\"     📈 Improvement vs baseline: {improvement:.2f}%\")\n",
        "\n",
        "best_combinations_df = pd.DataFrame(best_combinations)\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"SUMMARY TABLE:\")\n",
        "display(best_combinations_df)\n"
    ]))

    # Cell 11: Visualize Best Combinations
    cells.append(create_notebook_cell("code", [
        "# Visualize best combinations\n",
        "fig, ax = plt.subplots(figsize=(14, 7))\n",
        "\n",
        "x = range(len(best_combinations_df))\n",
        "bars = ax.bar(x, best_combinations_df['Cosine Similarity'], \n",
        "              color='forestgreen', alpha=0.8)\n",
        "\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels([f\"{row['Dataset']}\\n{row['Best Phase']} + {row['Best Model']}\" \n",
        "                     for _, row in best_combinations_df.iterrows()], \n",
        "                    rotation=45, ha='right')\n",
        "ax.set_ylabel('Cosine Similarity (PRIMARY METRIC)', fontweight='bold')\n",
        "ax.set_title('🏆 Best Combination (Phase + Model) per Dataset', fontsize=14, fontweight='bold')\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "\n",
        "for i, bar in enumerate(bars):\n",
        "    height = bar.get_height()\n",
        "    ax.text(bar.get_x() + bar.get_width()/2., height,\n",
        "            f'{height:.4f}\\n({best_combinations_df.iloc[i][\"Improvement vs Baseline (%)\"]}%)',\n",
        "            ha='center', va='bottom', fontsize=9)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    # Cell 12: Export Results
    cells.append(create_notebook_cell("code", [
        "# Export summary tables\n",
        "output_dir = RESULTS_BASE / '4_notebooks' / 'output'\n",
        "output_dir.mkdir(exist_ok=True)\n",
        "\n",
        "best_phases_df.to_csv(output_dir / 'best_phases_parsing.csv', index=False)\n",
        "best_models_df.to_csv(output_dir / 'best_models_per_phase_parsing.csv', index=False)\n",
        "best_combinations_df.to_csv(output_dir / 'best_combinations_parsing.csv', index=False)\n",
        "\n",
        "print(\"✅ Summary tables exported:\")\n",
        "print(f\"   - {output_dir / 'best_phases_parsing.csv'}\")\n",
        "print(f\"   - {output_dir / 'best_models_per_phase_parsing.csv'}\")\n",
        "print(f\"   - {output_dir / 'best_combinations_parsing.csv'}\")\n"
    ]))

    # Cell 13: Final Summary
    cells.append(create_notebook_cell("markdown", [
        "## 📊 Final Summary\n",
        "\n",
        "This notebook has answered the 3 key questions:\n",
        "\n",
        "### ✅ Question 1: Best Phase per Dataset\n",
        "Identified the optimal phase for each dataset based on Cosine Similarity (PRIMARY metric).\n",
        "\n",
        "### ✅ Question 2: Best Model per Phase per Dataset\n",
        "Identified the best-performing model for each dataset-phase combination.\n",
        "\n",
        "### ✅ Question 3: Best Combination per Dataset\n",
        "Identified the single best phase-model combination per dataset with improvement percentages.\n",
        "\n",
        "**Exported Files:**\n",
        "- `best_phases_parsing.csv` - Best phase per dataset\n",
        "- `best_models_per_phase_parsing.csv` - Best model for each dataset-phase\n",
        "- `best_combinations_parsing.csv` - Best phase-model per dataset\n",
        "\n",
        "**Next Steps:**\n",
        "- Review [qa_analysis.ipynb](qa_analysis.ipynb) for QA task analysis\n",
        "- Use results for production model selection\n",
        "- Refer to individual dataset notebooks in `by_dataset/` for detailed analysis\n"
    ]))

    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook

def create_qa_notebook():
    """Create qa_analysis.ipynb"""

    cells = []

    # Cell 1: Executive Summary
    cells.append(create_notebook_cell("markdown", [
        "# QA Tasks Aggregated Analysis\n",
        "\n",
        "## Executive Summary\n",
        "\n",
        "**Objective:** Aggregate and compare QA performance across all datasets to answer 3 key questions:\n",
        "\n",
        "### 🎯 Key Questions:\n",
        "1. **Which is the best phase for each dataset?**\n",
        "2. **Which is the best model per phase for each dataset?**\n",
        "3. **What's the best combination (phase + model) for each dataset?**\n",
        "\n",
        "**Datasets Included:**\n",
        "- ✅ **DocVQA_mini** (500 samples, 8 phases) - Document Visual QA\n",
        "- ✅ **InfographicVQA_mini** (500 samples, 11 phases) - Infographic QA\n",
        "- ⚠️ **dude_mini** (experimental) - Document Understanding\n",
        "- ⚠️ **chartqapro_mini** (experimental) - Chart QA\n",
        "\n",
        "**Evaluation Metrics:**\n",
        "- 🎯 **PRIMARY: GT in Pred** (Ground Truth in Prediction, higher is better)\n",
        "- **SECONDARY:** ANLS, Exact Match, Substring Match, Cosine Similarity\n",
        "\n",
        "**QA Strategies:**\n",
        "- **QA1 (OCR+VLM):** Two-step pipeline with OCR → LLM\n",
        "- **QA2 (VLM Parse+QA):** Single VLM does both parsing and QA\n",
        "- **QA3 (Direct VQA):** VLM sees image directly\n",
        "- **QA4 (Special):** Dataset-specific approaches\n"
    ]))

    # Cell 2: Imports
    cells.append(create_notebook_cell("code", [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import json\n",
        "import sys\n",
        "from pathlib import Path\n",
        "from typing import List, Dict, Optional\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "from tqdm.notebook import tqdm\n",
        "\n",
        "# Setup paths\n",
        "NOTEBOOK_DIR = Path.cwd()\n",
        "REPO_ROOT = NOTEBOOK_DIR.parent.parent.parent\n",
        "\n",
        "if str(REPO_ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(REPO_ROOT))\n",
        "\n",
        "# Import QA metrics\n",
        "from ocr_vs_vlm.metrics.evaluation_metrics import (\n",
        "    compute_anls,\n",
        "    compute_exact_match,\n",
        "    compute_substring_match,\n",
        "    compute_ground_truth_in_prediction,\n",
        "    compute_prediction_in_ground_truth\n",
        ")\n",
        "from ocr_vs_vlm.metrics.embedding_cache import EmbeddingCacheManager\n",
        "\n",
        "# Plotting style\n",
        "plt.style.use('seaborn-v0_8-whitegrid')\n",
        "plt.rcParams['figure.figsize'] = (14, 6)\n",
        "plt.rcParams['font.size'] = 11\n",
        "sns.set_palette('husl')\n",
        "\n",
        "# Paths\n",
        "RESULTS_BASE = REPO_ROOT / 'ocr_vs_vlm' / 'results'\n",
        "EMBEDDINGS_DIR = RESULTS_BASE / '3_embeddings'\n",
        "\n",
        "print(\"✅ Libraries loaded successfully!\")\n",
        "print(f\"📂 Results base: {RESULTS_BASE}\")\n"
    ]))

    # Cell 3: Dataset Configuration
    cells.append(create_notebook_cell("code", [
        "# QA Dataset configuration\n",
        "QA_DATASETS = {\n",
        "    'DocVQA_mini': {\n",
        "        'path': RESULTS_BASE / '2_clean' / 'DocVQA_mini',\n",
        "        'phases': ['QA1a', 'QA1b', 'QA1c', 'QA2a', 'QA2b', 'QA2c', 'QA3a', 'QA3b'],\n",
        "        'status': 'production'\n",
        "    },\n",
        "    'InfographicVQA_mini': {\n",
        "        'path': RESULTS_BASE / '2_clean' / 'InfographicVQA_mini',\n",
        "        'phases': ['QA1a', 'QA1b', 'QA1c', 'QA2a', 'QA2b', 'QA2c', 'QA3a', 'QA3b', 'QA4a', 'QA4b', 'QA4c'],\n",
        "        'status': 'production'\n",
        "    },\n",
        "    'dude_mini': {\n",
        "        'path': RESULTS_BASE / '2_clean' / 'dude_mini',\n",
        "        'phases': ['QA1a', 'QA1b', 'QA1c', 'QA2a', 'QA2b', 'QA2c', 'QA3a', 'QA3b'],\n",
        "        'status': 'experimental'\n",
        "    },\n",
        "    'chartqapro_mini': {\n",
        "        'path': RESULTS_BASE / '2_clean' / 'chartqapro_mini',\n",
        "        'phases': ['QA1a', 'QA1b', 'QA1c', 'QA2a', 'QA2b', 'QA2c', 'QA3a', 'QA3b'],\n",
        "        'status': 'experimental'\n",
        "    }\n",
        "}\n",
        "\n",
        "def get_phase_strategy(phase: str) -> str:\n",
        "    \"\"\"Get strategy group for phase\"\"\"\n",
        "    if phase.startswith('QA1'): return 'QA1 (OCR+VLM)'\n",
        "    elif phase.startswith('QA2'): return 'QA2 (VLM Parse+QA)'\n",
        "    elif phase.startswith('QA3'): return 'QA3 (Direct VQA)'\n",
        "    elif phase.startswith('QA4'): return 'QA4 (Special)'\n",
        "    return 'Unknown'\n",
        "\n",
        "print(\"📁 QA Dataset Configuration:\")\n",
        "for name, config in QA_DATASETS.items():\n",
        "    status_icon = '✅' if config['status'] == 'production' else '⚠️'\n",
        "    print(f\"  {status_icon} {name}: {len(config['phases'])} phases\")\n"
    ]))

    # Cell 4: Helper Functions
    cells.append(create_notebook_cell("code", [
        "def is_valid_row(row, pred_col: str, err_col: Optional[str] = None) -> bool:\n",
        "    \"\"\"Check if prediction is valid\"\"\"\n",
        "    pred_value = row[pred_col]\n",
        "    if pd.isna(pred_value) or str(pred_value).strip() == \"\":\n",
        "        return False\n",
        "    if err_col and err_col in row.index:\n",
        "        if pd.notna(row[err_col]) and str(row[err_col]).strip() != \"\":\n",
        "            return False\n",
        "    return True\n",
        "\n",
        "def parse_ground_truths(gt_string) -> List[str]:\n",
        "    \"\"\"Parse ground_truths from JSON string\"\"\"\n",
        "    if pd.isna(gt_string):\n",
        "        return []\n",
        "    if isinstance(gt_string, list):\n",
        "        return gt_string\n",
        "    try:\n",
        "        return json.loads(gt_string)\n",
        "    except:\n",
        "        return [str(gt_string)]\n",
        "\n",
        "def calculate_qa_metrics(pred: str, ground_truths: List[str], \n",
        "                         phase: str, sample_id: str, model: str,\n",
        "                         emb_manager: EmbeddingCacheManager) -> Dict:\n",
        "    \"\"\"Calculate all QA metrics\"\"\"\n",
        "    if pd.isna(pred) or pred == \"\" or not ground_truths:\n",
        "        return {\n",
        "            'gt_in_pred': 0.0,\n",
        "            'anls': 0.0,\n",
        "            'exact_match': 0.0,\n",
        "            'substring_match': 0.0,\n",
        "            'cosine_similarity': 0.0\n",
        "        }\n",
        "    \n",
        "    pred_str = str(pred)\n",
        "    cosine_sim = emb_manager.compute_cosine_similarity(\n",
        "        phase=phase, ground_truth=ground_truths[0],\n",
        "        prediction=pred_str, sample_id=sample_id, model=model\n",
        "    )\n",
        "    \n",
        "    return {\n",
        "        'gt_in_pred': compute_ground_truth_in_prediction(pred_str, ground_truths),\n",
        "        'anls': compute_anls(pred_str, ground_truths, threshold=0.5),\n",
        "        'exact_match': compute_exact_match(pred_str, ground_truths),\n",
        "        'substring_match': compute_substring_match(pred_str, ground_truths),\n",
        "        'cosine_similarity': float(cosine_sim)\n",
        "    }\n",
        "\n",
        "print(\"✅ Helper functions defined\")\n"
    ]))

    # Cell 5: Load Data
    cells.append(create_notebook_cell("code", [
        "# Load all QA datasets\n",
        "all_data = {}\n",
        "embedding_managers = {}\n",
        "\n",
        "for dataset_name, config in QA_DATASETS.items():\n",
        "    print(f\"\\nLoading {dataset_name}...\")\n",
        "    \n",
        "    # Initialize embedding manager\n",
        "    emb_manager = EmbeddingCacheManager(dataset_name, EMBEDDINGS_DIR)\n",
        "    embedding_managers[dataset_name] = emb_manager\n",
        "    \n",
        "    dataset_dfs = {}\n",
        "    for phase in config['phases']:\n",
        "        file_path = config['path'] / f\"{phase}.csv\"\n",
        "        if file_path.exists():\n",
        "            df = pd.read_csv(file_path)\n",
        "            dataset_dfs[phase] = df\n",
        "            print(f\"  ✅ {phase}: {len(df)} samples\")\n",
        "        else:\n",
        "            print(f\"  ⚠️  {phase}: File not found\")\n",
        "    \n",
        "    all_data[dataset_name] = {\n",
        "        'config': config,\n",
        "        'phase_dfs': dataset_dfs\n",
        "    }\n",
        "\n",
        "print(f\"\\n✅ Loaded {len(all_data)} datasets\")\n"
    ]))

    # Cell 6: Calculate Metrics
    cells.append(create_notebook_cell("code", [
        "# Calculate QA metrics for all datasets\n",
        "all_metrics = []\n",
        "\n",
        "for dataset_name, data in all_data.items():\n",
        "    if not data['phase_dfs']:\n",
        "        print(f\"⚠️  Skipping {dataset_name} - no data loaded\")\n",
        "        continue\n",
        "    \n",
        "    print(f\"\\nCalculating metrics for {dataset_name}...\")\n",
        "    emb_manager = embedding_managers[dataset_name]\n",
        "    \n",
        "    for phase, df in data['phase_dfs'].items():\n",
        "        pred_cols = [col for col in df.columns if col.startswith('prediction_')]\n",
        "        \n",
        "        for pred_col in pred_cols:\n",
        "            model = pred_col.replace('prediction_', '')\n",
        "            err_col = f'error_{model}'\n",
        "            \n",
        "            # Get valid rows\n",
        "            valid_rows = [r for _, r in df.iterrows() if is_valid_row(r, pred_col, err_col)]\n",
        "            \n",
        "            if not valid_rows:\n",
        "                continue\n",
        "            \n",
        "            # Calculate metrics\n",
        "            metrics_list = []\n",
        "            for row in tqdm(valid_rows, desc=f\"  {phase}/{model}\", leave=False):\n",
        "                gts = parse_ground_truths(row['ground_truths'])\n",
        "                m = calculate_qa_metrics(\n",
        "                    row[pred_col], gts, phase,\n",
        "                    row['sample_id'], model, emb_manager\n",
        "                )\n",
        "                metrics_list.append(m)\n",
        "            \n",
        "            # Aggregate\n",
        "            all_metrics.append({\n",
        "                'dataset': dataset_name,\n",
        "                'phase': phase,\n",
        "                'strategy': get_phase_strategy(phase),\n",
        "                'model': model,\n",
        "                'gt_in_pred': np.mean([m['gt_in_pred'] for m in metrics_list]),\n",
        "                'anls': np.mean([m['anls'] for m in metrics_list]),\n",
        "                'exact_match': np.mean([m['exact_match'] for m in metrics_list]),\n",
        "                'substring_match': np.mean([m['substring_match'] for m in metrics_list]),\n",
        "                'cosine_similarity': np.mean([m['cosine_similarity'] for m in metrics_list]),\n",
        "                'valid_samples': len(valid_rows)\n",
        "            })\n",
        "\n",
        "metrics_df = pd.DataFrame(all_metrics)\n",
        "print(f\"\\n✅ Calculated metrics for {len(metrics_df)} combinations\")\n"
    ]))

    # Cell 7: SECTION 3 - Best Phase per Dataset (KEY QUESTION #1)
    cells.append(create_notebook_cell("markdown", [
        "## 🎯 Section 3: Best Phase per Dataset\n",
        "\n",
        "**KEY QUESTION #1: Which is the best phase for each dataset?**\n",
        "\n",
        "This section identifies the best-performing phase for each dataset based on GT in Pred (PRIMARY metric).\n"
    ]))

    cells.append(create_notebook_cell("code", [
        "# Identify best phase per dataset\n",
        "print(\"=\"*80)\n",
        "print(\"BEST PHASE PER DATASET (by GT in Pred - PRIMARY METRIC)\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "best_phases = []\n",
        "\n",
        "for dataset in metrics_df['dataset'].unique():\n",
        "    dataset_data = metrics_df[metrics_df['dataset'] == dataset]\n",
        "    \n",
        "    # Average across models per phase\n",
        "    phase_avg = dataset_data.groupby('phase').agg({\n",
        "        'gt_in_pred': 'mean',\n",
        "        'anls': 'mean',\n",
        "        'exact_match': 'mean'\n",
        "    }).round(4)\n",
        "    \n",
        "    best_phase = phase_avg['gt_in_pred'].idxmax()\n",
        "    best_metrics = phase_avg.loc[best_phase]\n",
        "    \n",
        "    best_phases.append({\n",
        "        'Dataset': dataset,\n",
        "        'Best Phase': best_phase,\n",
        "        'Strategy': get_phase_strategy(best_phase),\n",
        "        'GT in Pred': best_metrics['gt_in_pred'],\n",
        "        'ANLS': best_metrics['anls'],\n",
        "        'Exact Match': best_metrics['exact_match']\n",
        "    })\n",
        "    \n",
        "    print(f\"\\n{dataset}:\")\n",
        "    print(f\"  🏆 Best Phase: {best_phase} ({get_phase_strategy(best_phase)})\")\n",
        "    print(f\"  🎯 GT in Pred: {best_metrics['gt_in_pred']:.4f}\")\n",
        "    print(f\"     ANLS: {best_metrics['anls']:.4f}, EM: {best_metrics['exact_match']:.4f}\")\n",
        "\n",
        "best_phases_df = pd.DataFrame(best_phases)\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"SUMMARY TABLE:\")\n",
        "display(best_phases_df)\n"
    ]))

    # Cell 8: Visualize Best Phases
    cells.append(create_notebook_cell("code", [
        "# Visualize best phase per dataset\n",
        "fig, ax = plt.subplots(figsize=(14, 6))\n",
        "\n",
        "x = range(len(best_phases_df))\n",
        "bars = ax.bar(x, best_phases_df['GT in Pred'], color='steelblue', alpha=0.8)\n",
        "\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels([f\"{row['Dataset']}\\n({row['Best Phase']})\" \n",
        "                     for _, row in best_phases_df.iterrows()], \n",
        "                    rotation=45, ha='right')\n",
        "ax.set_ylabel('GT in Pred (PRIMARY METRIC)', fontweight='bold')\n",
        "ax.set_title('🏆 Best Phase per Dataset', fontsize=14, fontweight='bold')\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "\n",
        "for i, bar in enumerate(bars):\n",
        "    height = bar.get_height()\n",
        "    ax.text(bar.get_x() + bar.get_width()/2., height,\n",
        "            f'{height:.4f}', ha='center', va='bottom')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    # Cell 9: SECTION 4 - Best Model per Phase per Dataset (KEY QUESTION #2)
    cells.append(create_notebook_cell("markdown", [
        "## 🎯 Section 4: Best Model per Phase per Dataset\n",
        "\n",
        "**KEY QUESTION #2: Which is the best model per phase for each dataset?**\n",
        "\n",
        "This section identifies the best-performing model for each dataset-phase combination.\n"
    ]))

    cells.append(create_notebook_cell("code", [
        "# Identify best model per phase per dataset\n",
        "print(\"=\"*80)\n",
        "print(\"BEST MODEL PER PHASE PER DATASET\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "best_models_per_phase = []\n",
        "\n",
        "for dataset in metrics_df['dataset'].unique():\n",
        "    print(f\"\\n{dataset}:\")\n",
        "    dataset_data = metrics_df[metrics_df['dataset'] == dataset]\n",
        "    \n",
        "    for phase in sorted(dataset_data['phase'].unique()):\n",
        "        phase_data = dataset_data[dataset_data['phase'] == phase]\n",
        "        best_idx = phase_data['gt_in_pred'].idxmax()\n",
        "        best = phase_data.loc[best_idx]\n",
        "        \n",
        "        best_models_per_phase.append({\n",
        "            'Dataset': dataset,\n",
        "            'Phase': phase,\n",
        "            'Strategy': best['strategy'],\n",
        "            'Best Model': best['model'],\n",
        "            'GT in Pred': best['gt_in_pred'],\n",
        "            'ANLS': best['anls'],\n",
        "            'Exact Match': best['exact_match']\n",
        "        })\n",
        "        \n",
        "        print(f\"  {phase}: {best['model'][:30]}... (GT in Pred={best['gt_in_pred']:.4f})\")\n",
        "\n",
        "best_models_df = pd.DataFrame(best_models_per_phase)\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"SUMMARY TABLE:\")\n",
        "display(best_models_df)\n"
    ]))

    # Cell 10: SECTION 5 - Best Combination per Dataset (KEY QUESTION #3)
    cells.append(create_notebook_cell("markdown", [
        "## 🎯 Section 5: Best Combination per Dataset\n",
        "\n",
        "**KEY QUESTION #3: What's the best combination (phase + model) for each dataset?**\n",
        "\n",
        "This section identifies the single best phase-model combination for each dataset.\n"
    ]))

    cells.append(create_notebook_cell("code", [
        "# Find best combination per dataset\n",
        "print(\"=\"*80)\n",
        "print(\"BEST COMBINATION (Phase + Model) PER DATASET\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "best_combinations = []\n",
        "\n",
        "for dataset in metrics_df['dataset'].unique():\n",
        "    dataset_data = metrics_df[metrics_df['dataset'] == dataset]\n",
        "    best_idx = dataset_data['gt_in_pred'].idxmax()\n",
        "    best = dataset_data.loc[best_idx]\n",
        "    \n",
        "    # Get baseline (QA1a) for comparison\n",
        "    baseline_data = dataset_data[dataset_data['phase'] == 'QA1a']\n",
        "    if len(baseline_data) > 0:\n",
        "        baseline_gt = baseline_data['gt_in_pred'].mean()\n",
        "        improvement = ((best['gt_in_pred'] - baseline_gt) / baseline_gt * 100) if baseline_gt > 0 else 0\n",
        "    else:\n",
        "        improvement = 0\n",
        "    \n",
        "    best_combinations.append({\n",
        "        'Dataset': dataset,\n",
        "        'Best Phase': best['phase'],\n",
        "        'Strategy': best['strategy'],\n",
        "        'Best Model': best['model'],\n",
        "        'GT in Pred': best['gt_in_pred'],\n",
        "        'ANLS': best['anls'],\n",
        "        'Exact Match': best['exact_match'],\n",
        "        'Improvement vs Baseline (%)': round(improvement, 2)\n",
        "    })\n",
        "    \n",
        "    print(f\"\\n{dataset}:\")\n",
        "    print(f\"  🏆 Best Combination: {best['phase']} + {best['model'][:40]}\")\n",
        "    print(f\"  🎯 GT in Pred: {best['gt_in_pred']:.4f}\")\n",
        "    print(f\"     ANLS: {best['anls']:.4f}, EM: {best['exact_match']:.4f}\")\n",
        "    if improvement != 0:\n",
        "        print(f\"     📈 Improvement vs baseline: {improvement:.2f}%\")\n",
        "\n",
        "best_combinations_df = pd.DataFrame(best_combinations)\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"SUMMARY TABLE:\")\n",
        "display(best_combinations_df)\n"
    ]))

    # Cell 11: Visualize Best Combinations
    cells.append(create_notebook_cell("code", [
        "# Visualize best combinations\n",
        "fig, ax = plt.subplots(figsize=(14, 7))\n",
        "\n",
        "x = range(len(best_combinations_df))\n",
        "bars = ax.bar(x, best_combinations_df['GT in Pred'], \n",
        "              color='forestgreen', alpha=0.8)\n",
        "\n",
        "ax.set_xticks(x)\n",
        "labels = []\n",
        "for _, row in best_combinations_df.iterrows():\n",
        "    model_short = row['Best Model'][:20] + '...' if len(row['Best Model']) > 20 else row['Best Model']\n",
        "    labels.append(f\"{row['Dataset']}\\n{row['Best Phase']} + {model_short}\")\n",
        "ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)\n",
        "ax.set_ylabel('GT in Pred (PRIMARY METRIC)', fontweight='bold')\n",
        "ax.set_title('🏆 Best Combination (Phase + Model) per Dataset', fontsize=14, fontweight='bold')\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "\n",
        "for i, bar in enumerate(bars):\n",
        "    height = bar.get_height()\n",
        "    improvement = best_combinations_df.iloc[i]['Improvement vs Baseline (%)']\n",
        "    ax.text(bar.get_x() + bar.get_width()/2., height,\n",
        "            f'{height:.4f}\\n({improvement}%)',\n",
        "            ha='center', va='bottom', fontsize=8)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    # Cell 12: Export Results
    cells.append(create_notebook_cell("code", [
        "# Export summary tables\n",
        "output_dir = RESULTS_BASE / '4_notebooks' / 'output'\n",
        "output_dir.mkdir(exist_ok=True)\n",
        "\n",
        "best_phases_df.to_csv(output_dir / 'best_phases_qa.csv', index=False)\n",
        "best_models_df.to_csv(output_dir / 'best_models_per_phase_qa.csv', index=False)\n",
        "best_combinations_df.to_csv(output_dir / 'best_combinations_qa.csv', index=False)\n",
        "\n",
        "print(\"✅ Summary tables exported:\")\n",
        "print(f\"   - {output_dir / 'best_phases_qa.csv'}\")\n",
        "print(f\"   - {output_dir / 'best_models_per_phase_qa.csv'}\")\n",
        "print(f\"   - {output_dir / 'best_combinations_qa.csv'}\")\n"
    ]))

    # Cell 13: Strategy Analysis
    cells.append(create_notebook_cell("markdown", [
        "## 📊 Strategy Analysis\n",
        "\n",
        "Compare QA strategies (QA1, QA2, QA3, QA4) across datasets.\n"
    ]))

    cells.append(create_notebook_cell("code", [
        "# Strategy comparison\n",
        "print(\"Strategy Performance Across Datasets:\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "strategy_summary = metrics_df.groupby(['dataset', 'strategy']).agg({\n",
        "    'gt_in_pred': 'mean',\n",
        "    'anls': 'mean',\n",
        "    'exact_match': 'mean'\n",
        "}).round(4)\n",
        "\n",
        "display(strategy_summary)\n",
        "\n",
        "# Overall strategy ranking\n",
        "overall_strategy = metrics_df.groupby('strategy').agg({\n",
        "    'gt_in_pred': 'mean',\n",
        "    'anls': 'mean',\n",
        "    'exact_match': 'mean'\n",
        "}).round(4).sort_values('gt_in_pred', ascending=False)\n",
        "\n",
        "print(\"\\nOverall Strategy Ranking:\")\n",
        "display(overall_strategy)\n"
    ]))

    # Cell 14: Final Summary
    cells.append(create_notebook_cell("markdown", [
        "## 📊 Final Summary\n",
        "\n",
        "This notebook has answered the 3 key questions:\n",
        "\n",
        "### ✅ Question 1: Best Phase per Dataset\n",
        "Identified the optimal phase for each dataset based on GT in Pred (PRIMARY metric).\n",
        "\n",
        "### ✅ Question 2: Best Model per Phase per Dataset\n",
        "Identified the best-performing model for each dataset-phase combination.\n",
        "\n",
        "### ✅ Question 3: Best Combination per Dataset\n",
        "Identified the single best phase-model combination per dataset with improvement percentages.\n",
        "\n",
        "**Exported Files:**\n",
        "- `best_phases_qa.csv` - Best phase per dataset\n",
        "- `best_models_per_phase_qa.csv` - Best model for each dataset-phase\n",
        "- `best_combinations_qa.csv` - Best phase-model per dataset\n",
        "\n",
        "**Key Findings:**\n",
        "- QA strategies show varying performance across datasets\n",
        "- Direct VQA (QA3) often performs well for visual reasoning\n",
        "- OCR+VLM pipeline (QA1) excels for text-heavy documents\n",
        "\n",
        "**Next Steps:**\n",
        "- Review [parsing_analysis.ipynb](parsing_analysis.ipynb) for parsing task analysis\n",
        "- Use results for production model selection\n",
        "- Refer to individual dataset notebooks in `by_dataset/` for detailed analysis\n"
    ]))

    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook

def main():
    """Generate both notebooks"""
    output_dir = Path("/Users/kenzabenkirane/Documents/GitHub/research-playground/ocr_vs_vlm/results/4_notebooks")

    print("Creating parsing_analysis.ipynb...")
    parsing_nb = create_parsing_notebook()
    parsing_path = output_dir / "parsing_analysis.ipynb"
    with open(parsing_path, 'w') as f:
        json.dump(parsing_nb, f, indent=2)
    print(f"✅ Created: {parsing_path}")

    print("\nCreating qa_analysis.ipynb...")
    qa_nb = create_qa_notebook()
    qa_path = output_dir / "qa_analysis.ipynb"
    with open(qa_path, 'w') as f:
        json.dump(qa_nb, f, indent=2)
    print(f"✅ Created: {qa_path}")

    print("\n" + "="*80)
    print("✅ Both notebooks created successfully!")
    print("="*80)
    print(f"\nParsing Analysis: {parsing_path}")
    print(f"QA Analysis: {qa_path}")
    print("\nThese notebooks answer 3 key questions for each dataset:")
    print("  1. Which is the best phase?")
    print("  2. Which is the best model per phase?")
    print("  3. What's the best combination (phase + model)?")

if __name__ == "__main__":
    main()
