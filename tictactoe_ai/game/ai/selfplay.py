import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .env import TicTacToeEnv
from .net import PolicyValueNet

def play_game(net, n, epsilon=0.1):
    env = TicTacToeEnv(n)
    traj = []
    while True:
        probs, v = net.predict(env)
        moves = env.legal_moves()
        if np.random.rand() < epsilon:
            move = moves[np.random.randint(len(moves))]
        else:
            flat = probs.flatten()
            move = np.unravel_index(np.random.choice(n*n, p=flat), (n, n))
        state = env.encode()
        player = env.player
        env.step(move)
        traj.append((state, probs.flatten(), player))
        if env.done:
            w = env.winner
            returns = []
            for _, _, p in traj:
                if w == 0:
                    returns.append(0.0)
                else:
                    returns.append(1.0 if w == p else -1.0)
            return [(s, pi, z) for (s, pi, _), z in zip(traj, returns)]

def selfplay_train(model_path, n=3, games=200, epochs=5, batch_size=64, lr=1e-3, epsilon=0.1):
    net = PolicyValueNet(n=n)
    if model_path.exists():
        net.load(model_path)
        net.model.train()
    opt = optim.Adam(net.model.parameters(), lr=lr)
    buffer = []
    for _ in range(games):
        buffer.extend(play_game(net, n, epsilon=epsilon))
    xs = torch.tensor(np.array([b[0] for b in buffer]), dtype=torch.float32, device=net.device)
    pis = torch.tensor(np.array([b[1] for b in buffer]), dtype=torch.float32, device=net.device)
    zs = torch.tensor(np.array([b[2] for b in buffer]), dtype=torch.float32, device=net.device)
    m = xs.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(m, device=net.device)
        for i in range(0, m, batch_size):
            b = idx[i:i+batch_size]
            x_b = xs[b]
            pi_b = pis[b]
            z_b = zs[b]
            logits, v_pred = net.model(x_b)
            p_loss = -(pi_b * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            v_loss = F.mse_loss(v_pred, z_b)
            loss = p_loss + v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.save(model_path)
    return net
