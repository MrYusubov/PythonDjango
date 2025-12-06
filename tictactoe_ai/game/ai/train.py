import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .env import TicTacToeEnv
from .minimax import minimax_policy, minimax_value
from .net import PolicyValueNet

def generate_minimax_dataset(num_states=5000):
    env = TicTacToeEnv(3)
    data = []
    for _ in range(num_states):
        env.reset()
        steps = np.random.randint(0, 6)
        for _ in range(steps):
            moves = env.legal_moves()
            if not moves or env.check_winner() != 0:
                break
            m = moves[np.random.randint(len(moves))]
            env.step(m)
        if env.check_winner() != 0 or not env.legal_moves():
            continue
        p = minimax_policy(env)
        v = minimax_value(env)
        data.append((env.encode(), p.flatten(), v))
    return data

def pretrain_3x3(model_path, epochs=10, batch_size=64, lr=1e-3):
    net = PolicyValueNet(n=3)
    opt = optim.Adam(net.model.parameters(), lr=lr)
    data = generate_minimax_dataset()
    xs = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32, device=net.device)
    ps = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32, device=net.device)
    vs = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32, device=net.device)
    n = xs.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=net.device)
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            x_b = xs[b]
            p_b = ps[b]
            v_b = vs[b]
            logits, v_pred = net.model(x_b)
            p_loss = -(p_b * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            v_loss = F.mse_loss(v_pred, v_b)
            loss = p_loss + v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.save(model_path)
    return net
