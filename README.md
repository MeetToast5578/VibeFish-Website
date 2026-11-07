# VibeFish Website

Play chess against every version of [VibeFish](https://github.com/MeetToast5578/VibeFish-ChessBot)
(v1–v4) in the browser. The networks run client-side with ONNX Runtime Web, so the site is
static: no server, no GPU, nothing to pay for.

## Run locally

```bash
npm start            # then open http://localhost:8000
```

## Publish

GitHub → Settings → Pages → Deploy from a branch → `main`, folder `/ (root)`.

## Files

- `index.html`: the page (board: chessground, rules: chess.js, inference: onnxruntime-web, all from jsDelivr).
- `engine.js`: the board encodings from the chessbot repo's `model.py` / `models/v1/model.py`, ported to JS, and the move choice (v4 blends in the action-value head, `--mix 2`, like `play.py`).
- `models/v1–v4.onnx`: exported from the `.pt` checkpoints. Weights stay fp16 and are cast to fp32 inside the graph, so moves match `play.py`.
- `tools/export.py`: re-export the models and the test fixture.
- `tests/`: `npm test` replays 335 positions recorded from the Python bot and checks that the JS encodings and every version's move are identical.

## Adding a new version (e.g. v4.1)

1. Add it to `VERSIONS` in `tools/export.py` and in `engine.js` (file, size, mix, blurb).
2. `python tools/export.py ../VibeFish-ChessBot` (needs `torch chess numpy onnx onnxruntime`).
3. `npm install && npm test`.
