import numpy as np

class TicTacToeEnv:
    def __init__(self, n=3):
        self.n = n
        self.reset()

    def reset(self):
        self.board = np.zeros((self.n, self.n), dtype=np.int8)
        self.player = 1
        self.done = False
        self.winner = 0
        return self.board

    def legal_moves(self):
        if self.done:
            return []
        coords = np.argwhere(self.board == 0)
        return [tuple(c) for c in coords]

    def step(self, move):
        if self.done:
            return self.board, 0.0, True
        r, c = move
        if self.board[r, c] != 0:
            self.done = True
            self.winner = -self.player
            return self.board, -1.0, True
        self.board[r, c] = self.player
        w = self.check_winner()
        if w != 0:
            self.done = True
            self.winner = w
            return self.board, 1.0, True
        if len(self.legal_moves()) == 0:
            self.done = True
            self.winner = 0
            return self.board, 0.0, True
        self.player *= -1
        return self.board, 0.0, False

    def check_winner(self):
        b = self.board
        n = self.n
        for i in range(n):
            s = b[i, :].sum()
            if abs(s) == n:
                return int(np.sign(s))
            s = b[:, i].sum()
            if abs(s) == n:
                return int(np.sign(s))
        d1 = np.diag(b).sum()
        if abs(d1) == n:
            return int(np.sign(d1))
        d2 = np.diag(np.fliplr(b)).sum()
        if abs(d2) == n:
            return int(np.sign(d2))
        return 0

    def canonical(self):
        if self.player == 1:
            return self.board.copy()
        return -self.board.copy()

    def encode(self):
        x = self.canonical()
        p1 = (x == 1).astype(np.float32)
        p2 = (x == -1).astype(np.float32)
        return np.stack([p1, p2], axis=0)
