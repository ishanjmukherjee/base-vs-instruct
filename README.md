# base vs instruct

Side-by-side demo: the same prompt sent to Gemma 4 26B-A4B base (raw document
continuation) and Gemma 4 26B-A4B-it (single user turn through the chat
template, thinking off). Both are 4-bit MLX quantizations of the same
pretrained model, and both sample with Google's published defaults from
`generation_config.json` (temperature 1.0, top_p 0.95, top_k 64).

```
uv run serve.py
open http://127.0.0.1:8800
```

The page loads at once. Each pane says "loading model…" until its worker is up,
which takes about 20 s from a warm disk.

- Typing in either pane updates both prompts.
- Enter runs (or re-runs) both. Shift+Enter inserts a newline. Esc stops.
- Each run samples fresh, so re-running gives a new completion.

Weights: the base model downloads from `mlx-community/gemma-4-26b-a4b-4bit`
into the Hugging Face cache. The instruct model is read from the LM Studio copy
at `~/.lmstudio/models/mlx-community/gemma-4-26b-a4b-it-4bit` if present
(override with `INSTRUCT_MODEL=...`), otherwise it downloads from
`mlx-community/gemma-4-26b-a4b-it-4bit`. Together they use about 30 GB of
unified memory, so quit LM Studio and other model servers before the talk.
`PORT=...` changes the page port; the workers use the next two ports.
