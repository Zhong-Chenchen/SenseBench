# Data Layout

`infer.py` now defaults to a unified JSONL question file:

```text
data/
  questions.jsonl
  images/...
```

Each JSONL row is one sample:

```json
{"id": "...", "images": ["images/a.png"], "question": "...", "answer": "A", "meta": {"image_count": "single", "task": "whether", "distortion_family": "blur", "distortion_type": "blur_gaussian", "comparison": "none"}}
```

Required top-level fields are `id`, `images`, `question`, `answer`, and `meta`. The evaluation/table code normalizes these metadata fields:

- `meta.image_count`: `single` or `multi`, mapped to bench split.
- `meta.task`: `whether`, `what`, `how`, or `description`.
- `meta.distortion_family` / `meta.distortion_type`: mapped to ability/task columns.
- `meta.comparison`: `none`, `intra`, or temporal variants, mapped to `pair_type`.

By default `SENSEBENCH_DATA_ROOT` is `./data`, and `infer.py` reads `./data/questions.jsonl`. You can override the file with `SENSEBENCH_QUESTIONS_JSONL=/path/to/questions.jsonl` or `python infer.py --input-jsonl /path/to/questions.jsonl`.

Image paths stored in JSONL are interpreted relative to `SENSEBENCH_DATA_ROOT`. For example, `images/a.png` resolves to `./data/images/a.png`.

The legacy directory layout is still supported with `python infer.py --output-format legacy`:

```text
data/
  bench/
    single/
      problems/{whether,what,how,description}/*.json
      images/...
    Multi/
      problems/{whether,what,how,description}/*.json
      images/...
  add/
    problems/...
    images/...
```
