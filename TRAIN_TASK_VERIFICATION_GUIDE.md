# Task Verification Training - README

## Summary

`train_task_verification.py` trains a **binary classification model** to predict whether a **cooking video contains execution errors**. The script uses **Leave-One-Recipe-Out (LORO) Cross-Validation** to ensure robust evaluation across different recipes.

**Key Features:**
- LORO CV: Tests on each recipe separately (prevents overfitting to specific recipes)
- Two model architectures: Transformer (recommended) or MLP (baseline)
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
  ├─ Create Model (Transformer or MLP)
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
**Architecture:** Transformer-only (no MLP baseline)
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
    'used_hiero': bool                          # Metadata flag
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
    --num_epochs 150 \
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
