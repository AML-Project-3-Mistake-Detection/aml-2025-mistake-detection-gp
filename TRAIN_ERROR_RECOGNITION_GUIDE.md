# Error Recognition Training - README

## Summary

Error Recognition is a **step-level classification task** that predicts whether a specific cooking step contains execution errors and classifies the error type (if present). Multiple model architectures are supported for flexible experimentation.

**Key Features:**
- Step-level binary classification: does this step have errors (yes/no)?
- Optional error category classification: what type of error occurred? (8 categories)
- Two model architectures: MLP (baseline) and Transformer (ErFormer)
- Flexible data splits: step-level or recording-level
- Two evaluation granularities: sub-step and step-level normalization
- BCEWithLogitsLoss with class weights to handle data imbalance

**Input:** Step-level features from video backbone (OmniVore or SlowFast)  
**Output:** Binary predictions of step correctness + optional error category labels

---

## File Flow Overview

```
START
  ↓
Parse Arguments (variant, backbone, split, modality, etc.)
  ↓
Load Data (CaptainCookStepDataset)
  ├─ Load step annotations (step_annotations.json)
  └─ Load error annotations (error_annotations.json)
  ↓
Create Model (MLP or ErFormer)
  ↓
Training Loop (with learning rate scheduler)
  ├─ For each epoch:
  │  ├─ train_epoch(): Forward, compute loss, backward, update weights
  │  ├─ Evaluate on val set
  │  ├─ Check if metrics improved
  │  └─ Save best model (if improved)
  └─ Early stopping via scheduler (ReduceLROnPlateau)
  ↓
Test Evaluation
  ├─ Load best model checkpoint
  ├─ Compute predictions on test set
  ├─ Aggregate metrics (sub-step and step level)
  └─ Compute PR-AUC and standard AUC
  ↓
Export Results (CSV file with per-fold results)
  ↓
END
```

---

## Function Purpose Mapping

| Function | Purpose | Called By |
|----------|---------|-----------|
| `fetch_model(config)` | Create appropriate model (MLP or ErFormer) based on config.variant | Training script |
| `train_epoch(model, device, train_loader, optimizer, epoch, criterion)` | Execute one training epoch: forward pass, compute loss, backward pass, gradient clipping, weight update | `train_model_base()` |
| `train_model_base(train_loader, val_loader, config, test_loader)` | Orchestrate entire training: model creation, optimizer setup, epoch loop, checkpoint saving, learning rate scheduling | Main entry point |
| `test_er_model(model, test_loader, criterion, device, ...)` | Evaluate model on test set: predictions, metrics (precision/recall/F1/AUC/PR-AUC), save results to CSV | Evaluation script |
| `save_results_to_csv(config, sub_step_metrics, step_metrics, ...)` | Export evaluation results to CSV file for aggregation | `test_er_model()` |
| `fetch_model_name(config)` | Generate standardized model checkpoint name | Training/evaluation |
| `store_model(model, config, ckpt_name)` | Save model weights to checkpoint file | Training loop |

---

## Detailed Component Descriptions

### Model Architectures

#### **MLP (Baseline)**
**Purpose:** Simple feedforward baseline for error recognition.

**Architecture:**
- Single hidden layer: Input → Linear(input_dim, 512) → ReLU → Linear(512, 1) → Sigmoid
- No sequence processing (treats all features as flat vector)
- Fastest to train

**Best for:** Quick baseline comparison, memory-constrained environments

---

#### **ErFormer (Transformer)**
**Purpose:*e processing with multi-head self-attention.* Transformer-based sequenc

**Architecture:**
- Input: Feature sequence (e.g., from OmniVore: [video_1024] or SlowFast: [video_400])
- Transformer Encoder: 1 layer of self-attention (8 heads)
- Global pooling: Extract aggregated output across time dimension
- MLP Decoder: Linear(1024 or 400, 512) → ReLU → Linear(512, 1) → Sigmoid

**Key Parameters:** `nhead=8`, `dim_feedforward=2048`, `num_layers=1`

**Best for:** Multi-modal features, capturing feature interactions

---



### Dataset: CaptainCookStepDataset

**Purpose:** Load and prepare step-level data from CaptainCook4D dataset.

**Key Properties:**
- Loads annotations from `step_annotations.json` and `error_annotations.json`
- Supports two splits:
  - **Step-level split**: Each step is independent sample
  - **Recording-level split**: Train/val/test grouped by video recording
- Handles video features: from OmniVore (1024-dim) or SlowFast (400-dim)
- Supports error category classification (8 categories)

**Error Categories (for error category recognition task):**

| Category | Label | Description |
|----------|-------|-------------|
| PREPARATION_ERROR | 0 | Incorrect ingredient preparation (chopping, mixing, etc.) |
| MEASUREMENT_ERROR | 1 | Wrong quantity or measurement |
| ORDER_ERROR | 2 | Wrong sequence of steps |
| TIMING_ERROR | 3 | Steps done at wrong time or for wrong duration |
| TECHNIQUE_ERROR | 4 | Incorrect cooking technique applied |
| TEMPERATURE_ERROR | 5 | Oven/cooking temperature incorrect |
| MISSING_STEP | 6 | A step was skipped |
| OTHER | 7 | Any other error type |

---

## Training Configuration

### Loss Function

```
BCEWithLogitsLoss(pos_weight=2.5)
```

**Why pos_weight=2.5?** Balances class imbalance (more non-error steps than error steps in dataset). Higher weight assigned to positive (error) class to increase recall.

### Optimizer

```
Adam(lr=0.001, weight_decay=0.0001)
```

### Learning Rate Scheduler

```
ReduceLROnPlateau(mode='max', factor=0.1, patience=5, threshold=1e-4)
```

**Strategy:** If validation metric doesn't improve for 5 evaluations, reduce learning rate by 10%. Helps fine-tune as training progresses.

### Gradient Clipping

```
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients during training.

---

## Evaluation Metrics

### Standard Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | % correct predictions |
| **Precision** | TP/(TP+FP) | Of predicted errors, what % are correct? |
| **Recall** | TP/(TP+FN) | Of actual errors, what % did we find? |
| **F1** | 2·(P·R)/(P+R) | Harmonic mean of precision & recall |
| **AUC** | Area under ROC curve | Ranking quality (0.5=random, 1.0=perfect) |
| **PR-AUC** | Area under Precision-Recall curve | Better for imbalanced data |

### Evaluation Granularities

**Sub-step Normalization:**
- Averages metrics across sub-step samples within each step
- Fine-grained evaluation

**Step Normalization:**
- Aggregates to step level (one prediction per step)
- Coarse-grained evaluation

Both are computed and reported separately.

---

## Data Splits

### Step-Level Split
```
Each unique step = one sample
Train: 70% of steps from all recipes
Val: 15% of steps from all recipes
Test: 15% of steps from all recipes
```

**Pro:** More samples, better generalization  
**Con:** Potential leakage (same recipe appears in train and test)

### Recording-Level Split
```
Load split configuration from file (e.g., recordings_data_split.json)
Train/Val/Test grouped by recording_id
Ensures complete recipes not mixed between splits
```

**Pro:** No recipe leakage  
**Con:** Fewer samples

---

## Usage Example

### Training Error Recognition (Binary Classification)

```bash
python base.py \
    --task_name ERROR_RECOGNITION \
    --variant TRANSFORMER \
    --backbone omnivore \
    --split recordings \
    --modality video \
    --num_epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --device cuda
```

### Error Category Recognition (Multi-class)

```bash
python base.py \
    --task_name ERROR_CATEGORY_RECOGNITION \
    --variant TRANSFORMER \
    --backbone omnivore \
    --split recordings \
    --modality video \
    --error_category TECHNIQUE_ERROR \
    --num_epochs 100 \
    --batch_size 32 \
    --device cuda
```

### Evaluation on Test Set

```bash
python core/evaluate.py \
    --split recordings \
    --backbone omnivore \
    --variant TRANSFORMER \
    --ckpt checkpoints/error_recognition_best/TRANSFORMER/omnivore/model.pt \
    --threshold 0.5
```

---

## Output Structure

**Checkpoint Saving:**
```
checkpoints/
  {task_name}/
    {variant}/
      {backbone}/
        {model_name}.pt
```

Example: `checkpoints/ERROR_RECOGNITION/TRANSFORMER/omnivore/error_recognition_ERROR_RECOGNITION_omnivore_TRANSFORMER_video.pt`

**Training Stats:**
```
stats/
  {task_name}/
    {variant}/
      {backbone}/
        {model_name}_training_performance.txt
```

**Results:**
```
results/
  {task_name}/
    combined_results/
      step_{step_norm}_substep_{substep_norm}_threshold_{threshold}.csv
```

---

## Related Files

- `base.py` - Entry point for training error recognition models
- `core/models/er_former.py` - ErFormer (Transformer) model definition
- `core/models/blocks.py` - Shared components (EncoderLayer, Encoder, MLP, fetch_input_dim)
- `dataloader/CaptainCookStepDataset.py` - Step-level dataset loader
- `core/evaluate.py` - Evaluation script for test set prediction

---

# APPENDIX: Component Summaries

This section summarizes the key components and their methods without full source code.

---

## A. ErFormer Model (`core/models/er_former.py`)

**Purpose:** Transformer-based model for step-level error recognition with multi-modal feature support.

**Architecture:**
- Transformer encoder to process feature sequences with self-attention
- Global pooling: aggregates features across time dimension
- MLP decoder for binary classification

**Key Parameters:** `nhead=8`, `dim_feedforward=2048`, `num_layers=1`

**Methods:**

| Method | Purpose |
|--------|---------|
| `__init__(config)` | Initialize transformer encoder and MLP decoder |
| `forward(input_data)` | Process features → transformer encoding → modality extraction → binary prediction |

---

## B. CaptainCookStepDataset (`dataloader/CaptainCookStepDataset.py`)

**Purpose:** PyTorch Dataset for step-level error recognition with support for multiple modalities and error categories.

**Data Format:**
- Loads from `step_annotations.json` and `error_annotations.json`
- Supports step-level or recording-level splits
- Handles multi-modal features: video, audio, text, depth
- Supports binary classification (error/no-error) and multi-class error category classification

**Key Features:**
- Error category mapping (8 categories)
- Flexible split mechanisms (step vs. recording)
- Phase support (train/val/test)

**Methods:**

| Method | Purpose |
|--------|---------|
| `__init__(config, phase, split)` | Load annotations, build error labels, initialize split |
| `__getitem__(idx)` | Return single sample (features, error label) or (features, error_category) |
| `__len__()` | Return total number of samples |
| `_build_error_category_label_name_map()` | Create bidirectional mapping between error names and numeric labels |
| `_build_error_category_labels()` | Parse error annotations and build label dictionary |
| `_init_step_split()` | Create step-level train/val/test split |
| `_init_other_split_from_file()` | Load recording-level split from JSON file |

---

## C. Blocks and Utilities (`core/models/blocks.py`)

**Purpose:** Shared components used across multiple model architectures.

**Key Components:**

| Component | Used By | Purpose |
|-----------|---------|---------|
| `EncoderLayer` | ErFormer | PyTorch TransformerEncoderLayer (alias) |
| `Encoder` | ErFormer | PyTorch TransformerEncoder (alias) |
| `MLP` | ErFormer, MLP | Simple feedforward network (2 layers) |
| `fetch_input_dim(config, decoder=False)` | All models | Return input dimension based on backbone + modalities |
| `PositionalEncoding` | Error Recognition | Sinusoidal positional encoding (if needed) |

**fetch_input_dim Logic:**
```
- OMNIVORE: 1024 (video features)
- SlowFast: 400 (video features)
```

---

## D. Training Flow (`base.py`)

**Purpose:** Main training orchestration for error recognition models.

**High-Level Steps:**

1. **Model Creation:** `fetch_model()` creates model based on config.variant
2. **Optimizer Setup:** Adam optimizer with weight decay
3. **Loss Function:** BCEWithLogitsLoss with pos_weight=2.5
4. **Scheduler:** ReduceLROnPlateau for adaptive learning rate
5. **Training Loop:** For each epoch:
   - `train_epoch()`: Forward → compute loss → backward → clip gradients → update weights
   - Validation: Compute metrics on val set
   - Learning rate update based on validation metric
   - Save best model checkpoint if improved
6. **Testing:** Load best checkpoint and evaluate on test set
7. **Results Export:** Save metrics to CSV file

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `fetch_model(config)` | Create appropriate model instance |
| `train_epoch()` | Execute one training epoch |
| `train_model_base()` | Main training orchestration |
| `test_er_model()` | Evaluate on test set and export results |
| `save_results_to_csv()` | Export metrics to CSV |
| `store_model()` | Save checkpoint to disk |

---

## E. Annotations Format

**Files:**
- `step_annotations.json` - Step boundaries and metadata
- `error_annotations.json` - Step-level error labels and categories

**Structure:**
```
step_annotations.json:
{
  "recording_id": {
    "steps": [
      {
        "step_id": int,
        "start_time": float,
        "end_time": float,
        "description": str
      },
      ...
    ]
  }
}

error_annotations.json:
[
  {
    "recording_id": str,
    "step_annotations": [
      {
        "step_id": int,
        "errors": [
          {
            "tag": str,  # "TechniqueError", "MeasurementError", etc.
            "description": str,
            "segment_frames": [start, end]
          }
        ]
      }
    ]
  }
]
```

---

## Summary of Key Differences from Task Verification

| Aspect | Task Verification | Error Recognition |
|--------|-------------------|-------------------|
| **Task Level** | Video-level | Step-level |
| **Output** | Binary (has errors: yes/no) | Binary + optional category (8 types) |
| **Models** | Transformer only | MLP, Transformer |
| **CV Strategy** | Leave-One-Recipe-Out (LORO) | Standard train/val/test split |
| **Input** | Step embeddings (1024D) | Raw features from backbone |
| **Loss** | BCEWithLogitsLoss (no weights) | BCEWithLogitsLoss (pos_weight=2.5) |
| **Metrics** | accuracy, precision, recall, F1, AUC | Same + PR-AUC, sub-step vs step granularities |
| **Data Splitting** | LORO (recipe-level) | Step-level or Recording-level |
