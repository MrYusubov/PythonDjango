import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PVModel(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.c1 = nn.Conv2d(2, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.p_head = nn.Conv2d(channels, 1, 1)
        self.v_head1 = nn.Conv2d(channels, 1, 1)
        self.v_fc1 = nn.Linear(1, channels)
        self.v_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        p = self.p_head(x)
        p = p.flatten(1)
        v = F.relu(self.v_head1(x))
        v = v.mean(dim=[2,3])
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return p, v.squeeze(1)

class PolicyValueNet:
    def __init__(self, n=3, device=None):
        self.n = n
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PVModel().to(self.device)

    def predict(self, env):
        x = torch.from_numpy(env.encode()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, v = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            v = float(v.cpu().numpy()[0])
        probs = probs.reshape(self.n, self.n)
        mask = (env.canonical() == 0).astype(np.float32)
        probs = probs * mask
        s = probs.sum()
        if s > 0:
            probs /= s
        else:
            moves = env.legal_moves()
            for m in moves:
                probs[m] = 1.0 / len(moves)
        return probs, v

    def best_move(self, env):
        probs, _ = self.predict(env)
        moves = env.legal_moves()
        best = max(moves, key=lambda m: probs[m])
        return list(best)

    def save(self, path):
        torch.save(self.model.state_dict(), str(path))

    def load(self, path):
        self.model.load_state_dict(torch.load(str(path), map_location=self.device))
        self.model.eval()
