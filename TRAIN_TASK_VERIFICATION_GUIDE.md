# Task Verification Training - README

## Summary

`train_task_verification.py` trains a **binary classification model** to predict whether a **cooking video contains execution errors**. The script uses **Leave-One-Recipe-Out (LORO) Cross-Validation** to ensure robust evaluation across different recipes.

**Key Features:**
- LORO CV: Tests on each recipe separately (prevents overfitting to specific recipes)
- Transformer architecture with self-attention mechanism
- Automatic early stopping based on F1 score
- Comprehensive visualization and metrics export (PNG, JSON, CSV)

**Input:** Step-level embeddings from video features  
**Output:** Predictions of video-level correctness (has errors: yes/no)

---

## File Flow Overview

```
START
  ↓
Parse Arguments
  ↓
Load Data (embeddings + annotations)
  ↓
Prepare Dataset (normalize, pad, create masks)
  ↓
LOOP: For Each Recipe (LORO CV)
  ├─ Split: Train on other recipes, Test on this one
  ├─ Create Model (Transformer encoder)
  ├─ Training Loop (with early stopping)
  ├─ Evaluate on test set
  └─ Save fold results
  ↓
Aggregate Results (compute mean/std of metrics)
  ↓
Visualize (create 4-panel plots)
  ↓
Export (PNG, JSON, CSV)
  ↓
END
```

---

## Function Purpose Mapping

| Function | Purpose | Called By |
|----------|---------|-----------|
| `load_precomputed_embeddings(npz_path, annotations_file)` | Load pre-extracted embeddings from NPZ file, normalize them, apply padding, create masks | `main()` |
| `get_recipe_groups(recording_ids, annotations_file)` | Group videos by recipe (for LORO CV split) | `main()` |
| `train_one_epoch(model, dataloader, criterion, optimizer, device)` | Train model for one epoch: forward pass, compute loss, backward pass, update weights | Main training loop |
| `evaluate(model, dataloader, device, threshold=0.5)` | Evaluate model: compute predictions and metrics (accuracy, precision, recall, F1, AUC) | Main training loop (validation) |
| `main()` | Orchestrate entire workflow: data loading → LORO CV loop → results aggregation → visualization | Script entry point |

**Note:** Task verification uses **Transformer-only** architecture. The `MLP` classes in `blocks.py` are for error recognition, not task verification.
---

## Detailed Function Descriptions

### `load_precomputed_embeddings(npz_path, annotations_file) → dict`

**Purpose:** Load and preprocess embedding data from a compressed NPZ file.

**Steps:**
1. Load embeddings dict from NPZ (format: `{video_id: embedding_array, ...}`)
2. Determine max steps and embedding dimension across all videos
3. Normalize embeddings using StandardScaler (per-video)
4. Pad all embeddings to same length (max_steps)
5. Create binary masks (True = real step, False = padding)
6. Load annotations and extract video-level labels (has_errors: 0/1)

**Returns:**
```python
{
    'embeddings': np.array (N_videos, max_steps, embedding_dim),
    'labels': np.array (N_videos,),            # 0=no errors, 1=has errors
    'masks': np.array (N_videos, max_steps),   # True=real, False=padding
    'recording_ids': list of str,               # Video IDs
}
```

---

### `get_recipe_groups(recording_ids, annotations_file) → dict`

**Purpose:** Group videos by their recipe for Leave-One-Recipe-Out split.

**Steps:**
1. Load annotations JSON
2. For each video ID, extract its activity_id (recipe)
3. Group videos under same recipe

**Returns:**
```python
{
    'recipe_id_1': {
        'name': 'Recipe Name',
        'recordings': [video_ids...]
    },
    'recipe_id_2': {...},
    ...
}
```

**Use:** Enables LORO CV by knowing which videos belong to each recipe.

---

### `train_one_epoch(model, dataloader, criterion, optimizer, device) → float`

**Purpose:** Execute one complete training epoch (update model weights once on all training data).

**Algorithm:**
```
1. Set model to training mode (activate dropout, batch norm)
2. For each batch:
   a. Extract embeddings, labels, masks
   b. Forward pass: outputs = model(embeddings, masks)
   c. Compute loss: loss = criterion(outputs, labels)
   d. Zero gradients: optimizer.zero_grad()
   e. Backward pass: loss.backward()
   f. Clip gradients: torch.nn.utils.clip_grad_norm_()
   g. Update weights: optimizer.step()
   h. Accumulate loss
3. Return average loss for epoch
```

**Returns:** Average loss value (float)

---

### `evaluate(model, dataloader, device, threshold=0.5) → tuple`

**Purpose:** Evaluate model performance on a dataset without updating weights.

**Algorithm:**
```
1. Set model to eval mode (disable dropout, use stored batch norm)
2. For each batch:
   a. Extract embeddings, labels, masks
   b. Forward pass: outputs = model(embeddings, masks)
   c. Convert to probabilities: probs = sigmoid(outputs)
   d. Apply threshold: preds = (probs > threshold).astype(int)
   e. Accumulate predictions, labels, probabilities
3. Compute metrics:
   - accuracy, precision, recall, f1_score, auc_roc
4. Return metrics and predictions
```

**Returns:**
```python
(
    metrics_dict,      # {'accuracy': float, 'precision': float, ...}
    predictions,       # np.array of 0/1
    labels,           # np.array of 0/1
    probabilities     # np.array of [0, 1]
)
```

---

### `main()`

**Purpose:** Orchestrate entire training pipeline using LORO CV.

**High-Level Steps:**

1. **Parse Arguments:** Load CLI arguments for model config, hyperparameters, paths
2. **Load Data:** Call `load_precomputed_embeddings()` or use `StepLocalizer`
3. **Create Dataset:** Instantiate `TaskVerificationDataset` with all videos
4. **Get Recipe Groups:** Call `get_recipe_groups()` for LORO split
5. **LORO CV Loop:** For each recipe as test set:
   - Split data: train = all other recipes, test = this recipe
   - Create fresh Transformer model
   - Training loop (up to num_epochs):
     - Call `train_one_epoch()` to train
     - Every 5 epochs: call `evaluate()` on both train and test
     - Check if F1 improved:
       - If yes: reset patience counter
       - If no: increment patience counter
       - If patience ≥ threshold: break (early stopping)
   - Save fold results (metrics, training history)
6. **Aggregate Results:** Compute mean/std of metrics across all folds
7. **Visualize:** Create 4-panel figure (boxplot, bar chart, learning curve, confusion matrix)
8. **Export:** Save PNG, JSON (detailed results), CSV (summary table)

**Key Hyperparameters:**
- `num_epochs`: Max training iterations per fold
- `batch_size`: Number of samples per batch
- `lr`: Learning rate for optimizer
- `patience`: Number of eval steps without improvement before early stopping
- `threshold`: Decision threshold for classification
- `dropout`: Dropout rate in model

---

## Validation Strategy: Leave-One-Recipe-Out (LORO) CV

**Why LORO?**

Standard train/val/test split can cause **recipe leakage**: the model learns recipe-specific patterns instead of general error detection.

LORO ensures each recipe is completely held out during training:

```
Recipe A training data (videos 1-10) ──┐
Recipe B training data (videos 11-20) ─┼─→ TRAIN ──→ Model
Recipe C training data (videos 21-30) ─┤
                                       
Recipe D test data (videos 31-40) ────────→ TEST ──→ F1, AUC, ...
```

This repeats for each recipe (D folds total if D recipes).

---

## Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | % correct predictions |
| **Precision** | TP/(TP+FP) | Of predicted errors, what % are correct? |
| **Recall** | TP/(TP+FN) | Of actual errors, what % did we find? |
| **F1** | 2·(P·R)/(P+R) | Harmonic mean of precision & recall |
| **AUC** | Area under ROC | Overall ranking quality (0.5=random, 1.0=perfect) |

Where: TP=True Positives, TN=True Negatives, FP=False Positives, FN=False Negatives

---

## Model Architecture: Transformer

```
Input: (batch_size, seq_len, embedding_dim)
  ↓
Transformer Encoder: Self-attention over steps
  ↓
Global Pooling: Attend to important steps
  ↓
Classification Head: MLP to binary prediction
  ↓
Output: (batch_size, 1) ∈ [0, 1]
```

**Why Transformer:**
- Self-attention learns which steps are critical for error detection
- Captures dependencies between steps
- State-of-the-art performance for sequence classification

---

## Output Files

**PNG**: `metrics_TIMESTAMP.png` - 4-panel visualization

**JSON**: `loro_cv_MODEL_TYPE_TIMESTAMP.json` - Complete results with config

**CSV**: `loro_cv_MODEL_TYPE_TIMESTAMP.csv` - Per-fold metrics summary

---

## Usage Example

```bash
python train_task_verification.py \
    --precomputed_embeddings "data/embeddings.npz" \
    --num_epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 10 \
    --threshold 0.57 \
    --device "cuda" \
    --save_dir "results/experiment_1"
```

---

## Related Files

- `core/models/task_verifier.py` - TaskVerifier model definition
- `dataloader/TaskVerificationDataset.py` - PyTorch Dataset class
- `extension/step_localization.py` - Feature extraction utilities (StepLocalizer)
- `TASK_VERIFICATION_README.md` - Additional documentation

---

# APPENDIX: Component Summaries

This section summarizes the key components and their methods without full source code.

---

## A. TaskVerifier Model (`core/models/task_verifier.py`)

**Purpose:** Transformer-based binary classification model that predicts whether a recipe execution video contains errors.

**Architecture:**
- Transformer encoder to process step sequences with self-attention
- Masked global pooling to aggregate step information (ignoring padding tokens)
- Binary classification head (MLP with ReLU and Sigmoid)

**Key Parameters:** `embedding_dim=1024`, `hidden_dim=512`, `num_heads=8`, `num_layers=2`, `dropout=0.7`

**Methods:**

| Method | Purpose |
|--------|---------|
| `__init__()` | Initialize transformer encoder and classification head |
| `forward(embeddings, mask)` | Process step embeddings → masked pooling → binary prediction (0-1) |
| `predict(embeddings, mask, threshold)` | Convert probabilities to binary predictions (0 or 1) |
| `get_num_parameters()` | Return count of trainable parameters |

---

## B. TaskVerificationDataset (`dataloader/TaskVerificationDataset.py`)

**Purpose:** PyTorch Dataset class for handling step-level embeddings with variable-length sequences and binary video-level labels.

**Data Format:**
- Input: Embeddings (N, max_steps, embed_dim), labels (N,), masks (N, max_steps), recording IDs
- Output per sample: Dictionary with embeddings, label, mask, recording_id
- Handles imbalanced data with class weight computation

**Key Attributes:** `num_samples`, `num_positive`, `num_negative`, `max_steps`, `embed_dim`

**Methods:**

| Method | Purpose |
|--------|---------|
| `__init__()` | Initialize with tensors and compute dataset statistics |
| `__len__()` | Return total number of samples |
| `__getitem__()` | Return single sample as dict (embeddings, label, mask, id) |
| `get_class_weights()` | Compute inverse frequency weights for imbalanced data |
| `get_statistics()` | Return dict with sample count, ratio, step ranges |
| `print_statistics()` | Print formatted dataset statistics to console |
| `collate_task_verification()` | Custom collate function for batch preparation |

---

## C. Step Localization (`extension/step_localization.py`)

## C. Step Localization (`extension/step_localization.py`)

**Purpose:** Extract step-level embeddings from video features using either ground truth or predicted boundaries.

**Two Routes:**

### Route A: StepLocalizer (Ground Truth Boundaries)

**Purpose:** Uses GT boundaries from annotations as upper-bound baseline.

**Methods:**

| Method | Purpose |
|--------|---------|
| `__init__()` | Load annotations and initialize feature directory |
| `process_video(recording_id)` | Process single video: load features, extract step embeddings using GT boundaries |
| `process_all_videos(recording_ids)` | Batch process multiple videos |
| `get_step_embeddings_matrix(video_data, pad_to_length)` | Return step embeddings as padded matrix with mask |
| `_extract_step_embedding(features, start, end)` | Average frame features within time boundaries |
| `_load_features(recording_id)` | Load EgoVLP .npz features for video |
| `_time_to_frame(time_sec)` | Convert seconds to frame index |

### Route B: PredictedBoundaryLocalizer (Predicted Boundaries)

**Purpose:** Uses predicted boundaries from temporal action detection models (e.g., ActionFormer) for end-to-end evaluation.

**Methods:**

| Method | Purpose |
|--------|---------|
| `__init__()` | Initialize with feature directory and optional predictions |
| `load_predictions_from_file(path)` | Load predictions from JSON file |
| `set_predictions(dict)` | Set predictions directly from dictionary |
| `process_video(recording_id, label)` | Process video using predicted boundaries + filtering |
| `process_all_videos(recording_ids, labels)` | Batch process multiple videos |
| `_filter_predictions(predictions)` | Apply confidence threshold, NMS, and max limit filters |
| `_nms(predictions)` | Non-Maximum Suppression to remove overlapping predictions |
| `_compute_iou(seg1, seg2)` | Compute Intersection-over-Union between temporal segments |
| `_extract_step_embedding(features, start, end)` | Average frame features within time boundaries |

### Utility Function:

**`prepare_dataset_for_task_verification(localizer, split_file, split)`**
- Purpose: Prepare padded step embeddings for task verification from split file
- Returns: Dict with embeddings (N, max_steps, 1024), labels (N,), masks (N, max_steps), recording_ids

### Data Flow:

```
ROUTE A (GT):
  Annotations (JSON) ──→ StepLocalizer ──→ Step embeddings
  Features (NPZ)    ──→                   (averaged over time bounds)
  
ROUTE B (Predicted):
  Predictions (JSON) ──→ PredictedBoundaryLocalizer ──→ Step embeddings
  Features (NPZ)    ──→ (+ NMS filtering)           (averaged over time bounds)
```

---

## D. Complete Step Annotations (`annotations/annotation_json/complete_step_annotations.json`)

**Purpose:** JSON file containing frame-by-frame annotations for all cooking videos with step boundaries and error labels.

**Top-Level Structure:**
- Keys: Recording IDs (e.g., "9_8", "1_7")
- Values: Video annotation objects with metadata and step array

**Video Annotation Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `recording_id` | str | Unique video identifier |
| `activity_id` | int | Recipe ID (groups videos by task type) |
| `activity_name` | str | Human-readable recipe name (e.g., "Ramen", "Microwave Egg Sandwich") |
| `person_id` | int | Which performer executed the task (for ablation studies) |
| `environment` | int | Kitchen setup ID (different kitchen appearances) |
| `steps` | list | Array of step objects |

**Step Object Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `step_id` | int | Action sequence number (1-12 depending on recipe) |
| `start_time` | float | Step start timestamp (seconds) |
| `end_time` | float | Step end timestamp (seconds) |
| `description` | str | Step instruction with verb prefix (e.g., "Pour-Pour 1 egg into...") |
| `has_errors` | bool | **GROUND TRUTH**: Does this step contain execution errors? |

**Video-Level Label Derivation:**

```
video_label = 1 if any(step['has_errors'] for step in steps) else 0
```

A video is labeled as **"has errors" (1)** if ANY of its steps contain errors; otherwise (0).

**Example Usage Pattern:**

1. Load video "9_8" from annotations
2. Extract activity_id=2 → "Ramen"
3. Read steps with their boundaries (start_time, end_time)
4. Check has_errors for each step
5. Derive video-level label: 1 if any step has error, else 0
6. Use recording_id to load corresponding embeddings from NPZ file
7. Pass to TaskVerificationDataset as input with label
