import numpy as np
from .env import TicTacToeEnv

def minimax_value(env):
    w = env.check_winner()
    if w != 0:
        return w * env.player
    moves = env.legal_moves()
    if not moves:
        return 0
    best = -2
    for m in moves:
        e2 = TicTacToeEnv(env.n)
        e2.board = env.board.copy()
        e2.player = env.player
        e2.done = env.done
        e2.winner = env.winner
        e2.step(m)
        v = -minimax_value(e2)
        if v > best:
            best = v
        if best == 1:
            break
    return best

def minimax_policy(env):
    moves = env.legal_moves()
    vals = []
    for m in moves:
        e2 = TicTacToeEnv(env.n)
        e2.board = env.board.copy()
        e2.player = env.player
        e2.step(m)
        vals.append(-minimax_value(e2))
    best = max(vals)
    probs = np.zeros((env.n, env.n), dtype=np.float32)
    for m, v in zip(moves, vals):
        if v == best:
            probs[m] = 1.0
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs
