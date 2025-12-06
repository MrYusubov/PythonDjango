import json
import numpy as np
from django.http import JsonResponse, HttpResponseNotAllowed
from django.conf import settings
from .ai.env import TicTacToeEnv
from .ai.net import PolicyValueNet

def health(request):
    return JsonResponse({"ok": True})

def ai_move(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    payload = json.loads(request.body.decode("utf-8"))
    board = np.array(payload["board"], dtype=np.int8)
    player = int(payload.get("player", 1))
    n = int(payload.get("n", board.shape[0]))
    env = TicTacToeEnv(n=n)
    env.board = board.copy()
    env.player = player
    model_path = settings.AI_MODELS_DIR / f"ttt_{n}x{n}.pt"
    if not model_path.exists():
        model_path = settings.AI_MODELS_DIR / "ttt_3x3.pt"
    net = PolicyValueNet(n=n)
    net.load(model_path)
    move = net.best_move(env)
    return JsonResponse({"move": move})
