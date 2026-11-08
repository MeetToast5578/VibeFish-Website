"""Export VibeFish checkpoints to ONNX for the website, plus a test fixture.

  python tools/export.py ../VibeFish-ChessBot

Needs torch, chess, numpy, onnx (and onnxruntime for the check). Writes models/v1..v4.1.onnx and
tests/fixture.json. Weights are stored as fp16 (as in the .pt files) and cast to fp32 inside the
graph, so the browser computes exactly what play.py computes, at half the download.
"""
import importlib.util
import json
import os
import random
import sys

import chess
import numpy as np
import onnx
import onnxruntime
import torch
from onnx import helper, numpy_helper

REPO = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "../VibeFish-ChessBot")
OUT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
import model as m2  # noqa: E402  (the chessbot repo's model.py: v2-v4)

spec = importlib.util.spec_from_file_location("v1model", os.path.join(REPO, "models/v1/model.py"))
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)

torch.backends.mha.set_fastpath_enabled(False)   # v1 uses nn.TransformerEncoder; the fused path won't export
torch.set_num_threads(1)


class V1(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):            # x: int64 [1, 66] tokens -> policy [1, 4096]
        return self.net(x)[0]


class V2(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):            # x: int64 [1, 68, 7] features -> policy, action [1, 4096]
        p, a, _ = self.net(x)
        return p, a


def fp16_weights(path):
    """Store float initializers as fp16 and cast back to fp32 at graph start."""
    g = onnx.load(path).graph
    casts = []
    for init in g.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        name = init.name
        init.CopyFrom(numpy_helper.from_array(arr, name + "_fp16"))
        casts.append(helper.make_node("Cast", [name + "_fp16"], [name], to=onnx.TensorProto.FLOAT))
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    model = onnx.load(path)
    model.graph.CopyFrom(g)
    onnx.checker.check_model(model)
    onnx.save(model, path)


VERSIONS = {}
v1 = m1.ChessGPT()
v1.load_state_dict(torch.load(os.path.join(REPO, "models/v1/model.pt"), map_location="cpu"))
v1.load_state_dict({k: v.half().float() for k, v in v1.state_dict().items()})  # match the fp16 export
VERSIONS["v1"] = v1.eval()
for v in ("v2", "v3", "v4", "v4.1"):
    VERSIONS[v] = m2.load(os.path.join(REPO, f"models/vibefish_{v.replace('.', '_')}.pt"))
MIX = {"v1": 0, "v2": 0, "v3": 0, "v4": 2, "v4.1": 2}    # README: v1-v3 policy only, v4 and v4.1 play with --mix 2

for v, net in VERSIONS.items():
    path = os.path.join(OUT, "models", f"{v}.onnx")
    if v == "v1":
        x = torch.tensor([m1.encode(chess.Board())])
        torch.onnx.export(V1(net), (x,), path, input_names=["x"], output_names=["policy"], opset_version=17, dynamo=False)
    else:
        x = torch.from_numpy(m2.encode(chess.Board()))[None].long()
        torch.onnx.export(V2(net), (x,), path, input_names=["x"], output_names=["policy", "action"], opset_version=17, dynamo=False)
    fp16_weights(path)
    print(v, f"{os.path.getsize(path) / 1e6:.1f} MB")


def py_pick(v, board):
    if v == "v1":
        return VERSIONS[v].pick(board)
    return VERSIONS[v].pick(board, mix=MIX[v])


# Fixture: positions with full history, the encodings and every version's move.
rng = random.Random(0)
games = []
scripted = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "d7d5", "e4e5", "f7f5"]
for g in range(6):
    board, moves = chess.Board(), []
    for ply in range(70):
        if board.is_game_over():
            break
        if g == 0 and ply < len(scripted):
            mv = chess.Move.from_uci(scripted[ply])     # repetitions + a legal en passant
        elif rng.random() < 0.5:
            mv = py_pick("v4", board)
        else:
            mv = rng.choice(list(board.legal_moves))
        board.push(mv)
        moves.append(mv.uci())
    games.append(moves)

positions = []
for moves in games:
    board = chess.Board()
    for i in range(len(moves) + 1):
        if not board.is_game_over():
            positions.append({
                "moves": moves[:i],
                "enc1": m1.encode(board),
                "enc2": m2.encode(board).flatten().tolist(),
                "picks": {v: py_pick(v, board).uci() for v in VERSIONS},
            })
        if i < len(moves):
            board.push_uci(moves[i])

# check the ONNX files reproduce the picks
for v in VERSIONS:
    sess = onnxruntime.InferenceSession(os.path.join(OUT, "models", f"{v}.onnx"))
    for pos in positions[:40]:
        board = chess.Board()
        for mv in pos["moves"]:
            board.push_uci(mv)
        legal = m2.legal_indices(board)
        if v == "v1":
            pol = sess.run(None, {"x": np.array([pos["enc1"]], np.int64)})[0][0]
            s = pol[legal]
        else:
            pol, act = sess.run(None, {"x": np.array(pos["enc2"], np.int64).reshape(1, 68, 7)})
            lp = pol[0][legal] - np.log(np.exp(pol[0][legal] - pol[0][legal].max()).sum()) - pol[0][legal].max()
            s = lp + MIX[v] * -np.log1p(np.exp(-act[0][legal]))
        assert m2.unorient_index(board, legal[int(np.argmax(s))]).uci() == pos["picks"][v], (v, pos["moves"])
os.makedirs(os.path.join(OUT, "tests"), exist_ok=True)
with open(os.path.join(OUT, "tests", "fixture.json"), "w") as f:
    json.dump(positions, f, separators=(",", ":"))
print(len(positions), "fixture positions; ONNX picks match PyTorch")
