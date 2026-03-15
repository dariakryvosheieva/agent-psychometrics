## Training an IRT model (and exporting question difficulties)

### 1) Data format
`py_irt` expects a **JSON Lines** (`.jsonlines`) file where each line is one subject:

- `subject_id`: string identifier for a learner/user
- `responses`: mapping from question/item id to a numeric response (usually `0`/`1`)

Example (two subjects, three questions):

```json
{"subject_id": "u1", "responses": {"q1": 1, "q2": 0, "q3": 1}}
{"subject_id": "u2", "responses": {"q1": 0, "q2": 0, "q3": 1}}
```

### 2) Install dependencies
From this folder’s parent directory:

```bash
cd /path/to/model_irt
python -m pip install -r py_irt/requirements.txt
```

### 3) Train
Train a 1PL model (difficulty only) and write outputs to `out/`:

```bash
python -m py_irt.cli train 1pl /ABS/PATH/data.jsonlines /ABS/PATH/out --device cpu
```

This produces:
- `out/parameters.json` (final epoch)
- `out/best_parameters.json` (best loss seen during training)

### 4) Export question difficulty dataset
`best_parameters.json` contains `diff` (difficulty) aligned to `item_ids` (index → question_id).

If your environment has `typer` installed:

```bash
python -m py_irt.cli export-question-difficulties /ABS/PATH/out/best_parameters.json /ABS/PATH/out/question_difficulties.csv
```

If you want a **stdlib-only** exporter (no `typer/pyro/torch` needed), use:

```bash
python py_irt/export_difficulties.py /ABS/PATH/out/best_parameters.json /ABS/PATH/out/question_difficulties.csv
```




