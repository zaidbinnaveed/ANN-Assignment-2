import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# TOY DATASET GENERATION
# ============================================================
# Toy Problem 1: f(x,y,z,t) = x^2 + exp(y) + z + sin(t)
# x -> convex, y -> monotone+convex, z -> monotone, t -> arbitrary
# Toy Problem 2: f(x,y) = x^2 * y^2
# x -> convex, y -> convex

def lhs_sample(n, d, lo, hi, seed=0):
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    s = sampler.random(n)
    return qmc.scale(s, lo, hi)

def generate_toy1(n_train=100, n_test=500):
    train = lhs_sample(n_train, 4, [0.0]*4, [4.0]*4, seed=1)
    test  = lhs_sample(n_test,  4, [0.0]*4, [4.0]*4, seed=2)
    def f(X):
        x,y,z,t = X[:,0], X[:,1], X[:,2], X[:,3]
        return (x**2 + np.exp(y) + z + np.sin(t)).reshape(-1,1)
    return train, f(train), test, f(test)

def generate_toy2(n_train=100, n_test=500):
    train = lhs_sample(n_train, 2, [0.0]*2, [4.0]*2, seed=3)
    test  = lhs_sample(n_test,  2, [0.0]*2, [4.0]*2, seed=4)
    def f(X):
        x,y = X[:,0], X[:,1]
        return (x**2 * y**2).reshape(-1,1)
    return train, f(train), test, f(test)

X1_train, y1_train, X1_test, y1_test = generate_toy1()
X2_train, y2_train, X2_test, y2_test = generate_toy2()
np.save('toy1_train.npy', np.hstack([X1_train, y1_train]))
np.save('toy1_test.npy',  np.hstack([X1_test,  y1_test]))
np.save('toy2_train.npy', np.hstack([X2_train, y2_train]))
np.save('toy2_test.npy',  np.hstack([X2_test,  y2_test]))
print("Datasets saved.")

# ============================================================
# PYTORCH IMPLEMENTATION
# ============================================================
# ISNN Architecture:
# Three branches for three input types:
#   Branch C  (convex):          weights >= 0, softplus activation
#   Branch MC (monotone-convex): weights >= 0, softplus activation
#   Branch M  (monotone):        weights >= 0, softplus activation
#   Branch A  (arbitrary):       unconstrained, tanh activation
# Output = linear combination via non-negative weights

class ISNN1_PyTorch(nn.Module):
    """
    ISNN-1 for Toy Problem 1: inputs x(convex), y(mono+convex), z(mono), t(arb)
    Architecture: 4 branches merged with non-negative output weights.
    Softplus used to enforce non-negativity of weights via exp parameterization.
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.hidden = hidden

        # Raw weight parameters (unconstrained; apply softplus to get positive weights)
        # Branch C: x -> convex (non-neg weights, softplus act)
        self.Wc1_raw = nn.Parameter(torch.randn(1, hidden) * 0.1)
        self.bc1     = nn.Parameter(torch.zeros(hidden))
        self.Wc2_raw = nn.Parameter(torch.randn(hidden, hidden) * 0.1)
        self.bc2     = nn.Parameter(torch.zeros(hidden))
        self.Wco_raw = nn.Parameter(torch.randn(hidden, 1) * 0.1)

        # Branch MC: y -> mono+convex (non-neg weights, softplus act)
        self.Wmc1_raw = nn.Parameter(torch.randn(1, hidden) * 0.1)
        self.bmc1     = nn.Parameter(torch.zeros(hidden))
        self.Wmc2_raw = nn.Parameter(torch.randn(hidden, hidden) * 0.1)
        self.bmc2     = nn.Parameter(torch.zeros(hidden))
        self.Wmco_raw = nn.Parameter(torch.randn(hidden, 1) * 0.1)

        # Branch M: z -> monotone (non-neg weights, softplus act)
        self.Wm1_raw = nn.Parameter(torch.randn(1, hidden) * 0.1)
        self.bm1     = nn.Parameter(torch.zeros(hidden))
        self.Wm2_raw = nn.Parameter(torch.randn(hidden, hidden) * 0.1)
        self.bm2     = nn.Parameter(torch.zeros(hidden))
        self.Wmo_raw = nn.Parameter(torch.randn(hidden, 1) * 0.1)

        # Branch A: t -> arbitrary (unconstrained, tanh)
        self.Wa1 = nn.Linear(1, hidden)
        self.Wa2 = nn.Linear(hidden, hidden)
        self.Wao = nn.Linear(hidden, 1)

        # Bias for output
        self.bias_out = nn.Parameter(torch.zeros(1))

    def softplus(self, x):
        return torch.log(1 + torch.exp(x))

    def pos(self, W_raw):
        return self.softplus(W_raw)

    def branch_convex(self, x, W1r, b1, W2r, b2, Wor):
        W1 = self.pos(W1r); W2 = self.pos(W2r); Wo = self.pos(Wor)
        h1 = self.softplus(x @ W1 + b1)
        h2 = self.softplus(h1 @ W2 + b2)
        return h2 @ Wo

    def branch_arb(self, x):
        h1 = torch.tanh(self.Wa1(x))
        h2 = torch.tanh(self.Wa2(h1))
        return self.Wao(h2)

    def forward(self, X):
        xc  = X[:, 0:1]
        xmc = X[:, 1:2]
        xm  = X[:, 2:3]
        xa  = X[:, 3:4]

        out_c  = self.branch_convex(xc,  self.Wc1_raw,  self.bc1,  self.Wc2_raw,  self.bc2,  self.Wco_raw)
        out_mc = self.branch_convex(xmc, self.Wmc1_raw, self.bmc1, self.Wmc2_raw, self.bmc2, self.Wmco_raw)
        out_m  = self.branch_convex(xm,  self.Wm1_raw,  self.bm1,  self.Wm2_raw,  self.bm2,  self.Wmo_raw)
        out_a  = self.branch_arb(xa)

        return out_c + out_mc + out_m + out_a + self.bias_out


class ISNN2_PyTorch(nn.Module):
    """
    ISNN-2 for Toy Problem 2: inputs x(convex), y(convex)
    Two convex branches combined with non-negative weights.
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.hidden = hidden

        # Branch C1: x -> convex
        self.Wc1_raw = nn.Parameter(torch.randn(1, hidden) * 0.1)
        self.bc1     = nn.Parameter(torch.zeros(hidden))
        self.Wc2_raw = nn.Parameter(torch.randn(hidden, hidden) * 0.1)
        self.bc2     = nn.Parameter(torch.zeros(hidden))
        self.Wco_raw = nn.Parameter(torch.randn(hidden, 1) * 0.1)

        # Branch C2: y -> convex
        self.Wc21_raw = nn.Parameter(torch.randn(1, hidden) * 0.1)
        self.bc21     = nn.Parameter(torch.zeros(hidden))
        self.Wc22_raw = nn.Parameter(torch.randn(hidden, hidden) * 0.1)
        self.bc22     = nn.Parameter(torch.zeros(hidden))
        self.Wc2o_raw = nn.Parameter(torch.randn(hidden, 1) * 0.1)

        self.bias_out = nn.Parameter(torch.zeros(1))

    def softplus(self, x):
        return torch.log(1 + torch.exp(x))

    def pos(self, W_raw):
        return self.softplus(W_raw)

    def branch_convex(self, x, W1r, b1, W2r, b2, Wor):
        W1 = self.pos(W1r); W2 = self.pos(W2r); Wo = self.pos(Wor)
        h1 = self.softplus(x @ W1 + b1)
        h2 = self.softplus(h1 @ W2 + b2)
        return h2 @ Wo

    def forward(self, X):
        xc1 = X[:, 0:1]
        xc2 = X[:, 1:2]
        out_c1 = self.branch_convex(xc1, self.Wc1_raw, self.bc1, self.Wc2_raw, self.bc2, self.Wco_raw)
        out_c2 = self.branch_convex(xc2, self.Wc21_raw, self.bc21, self.Wc22_raw, self.bc22, self.Wc2o_raw)
        return out_c1 + out_c2 + self.bias_out


def train_pytorch(model, X_tr, y_tr, X_te, y_te, epochs=2000, lr=1e-3):
    Xt  = torch.tensor(X_tr, dtype=torch.float32)
    yt  = torch.tensor(y_tr, dtype=torch.float32)
    Xte = torch.tensor(X_te, dtype=torch.float32)
    yte = torch.tensor(y_te, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(Xt)
        loss = nn.MSELoss()(pred, yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_te = model(Xte)
            loss_te = nn.MSELoss()(pred_te, yte)

        train_losses.append(loss.item())
        test_losses.append(loss_te.item())

        if (ep+1) % 500 == 0:
            print(f"  Epoch {ep+1:4d} | Train MSE: {loss.item():.6f} | Test MSE: {loss_te.item():.6f}")

    return train_losses, test_losses


print("\n=== PyTorch: ISNN-1 on Toy Problem 1 ===")
model1_pt = ISNN1_PyTorch(hidden=16)
tr1_pt, te1_pt = train_pytorch(model1_pt, X1_train, y1_train, X1_test, y1_test, epochs=3000)

print("\n=== PyTorch: ISNN-2 on Toy Problem 2 ===")
model2_pt = ISNN2_PyTorch(hidden=16)
tr2_pt, te2_pt = train_pytorch(model2_pt, X2_train, y2_train, X2_test, y2_test, epochs=3000)


# ============================================================
# MANUAL NUMPY IMPLEMENTATION WITH BACKPROPAGATION
# ============================================================
# softplus: sp(x) = log(1+exp(x)), sp'(x) = sigmoid(x)
# All constrained weights stored raw; positive via softplus

def softplus(x):
    return np.log1p(np.exp(np.clip(x, -500, 500)))

def softplus_grad(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def tanh_act(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1.0 - np.tanh(x)**2

class ISNN1_NumPy:
    """
    ISNN-1 manual implementation using matrix multiplication + backprop.
    Three constrained branches (C, MC, M) + one arbitrary branch (A).
    """
    def __init__(self, hidden=16, lr=1e-3):
        self.lr = lr
        h = hidden
        scale = 0.1

        # Branch C (convex) - raw params; positive weights = softplus(raw)
        self.Wc1r = np.random.randn(1, h) * scale
        self.bc1  = np.zeros((1, h))
        self.Wc2r = np.random.randn(h, h) * scale
        self.bc2  = np.zeros((1, h))
        self.Wcor = np.random.randn(h, 1) * scale

        # Branch MC (monotone+convex)
        self.Wmc1r = np.random.randn(1, h) * scale
        self.bmc1  = np.zeros((1, h))
        self.Wmc2r = np.random.randn(h, h) * scale
        self.bmc2  = np.zeros((1, h))
        self.Wmcor = np.random.randn(h, 1) * scale

        # Branch M (monotone)
        self.Wm1r = np.random.randn(1, h) * scale
        self.bm1  = np.zeros((1, h))
        self.Wm2r = np.random.randn(h, h) * scale
        self.bm2  = np.zeros((1, h))
        self.Wmor = np.random.randn(h, 1) * scale

        # Branch A (arbitrary) - unconstrained
        self.Wa1 = np.random.randn(1, h) * scale
        self.ba1 = np.zeros((1, h))
        self.Wa2 = np.random.randn(h, h) * scale
        self.ba2 = np.zeros((1, h))
        self.Wao = np.random.randn(h, 1) * scale
        self.bao = np.zeros((1, 1))

        self.bias = np.zeros((1, 1))

        # Adam optimizer state
        self.t = 0
        self.params = self._get_params()
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def _get_params(self):
        return [self.Wc1r, self.bc1, self.Wc2r, self.bc2, self.Wcor,
                self.Wmc1r, self.bmc1, self.Wmc2r, self.bmc2, self.Wmcor,
                self.Wm1r, self.bm1, self.Wm2r, self.bm2, self.Wmor,
                self.Wa1, self.ba1, self.Wa2, self.ba2, self.Wao, self.bao,
                self.bias]

    def forward(self, X):
        xc  = X[:, 0:1]
        xmc = X[:, 1:2]
        xm  = X[:, 2:3]
        xa  = X[:, 3:4]
        n = X.shape[0]

        # Branch C
        Wc1 = softplus(self.Wc1r); Wc2 = softplus(self.Wc2r); Wco = softplus(self.Wcor)
        zc1 = xc @ Wc1 + self.bc1
        hc1 = softplus(zc1)
        zc2 = hc1 @ Wc2 + self.bc2
        hc2 = softplus(zc2)
        outc = hc2 @ Wco

        # Branch MC
        Wmc1 = softplus(self.Wmc1r); Wmc2 = softplus(self.Wmc2r); Wmco = softplus(self.Wmcor)
        zmc1 = xmc @ Wmc1 + self.bmc1
        hmc1 = softplus(zmc1)
        zmc2 = hmc1 @ Wmc2 + self.bmc2
        hmc2 = softplus(zmc2)
        outmc = hmc2 @ Wmco

        # Branch M
        Wm1 = softplus(self.Wm1r); Wm2 = softplus(self.Wm2r); Wmo = softplus(self.Wmor)
        zm1 = xm @ Wm1 + self.bm1
        hm1 = softplus(zm1)
        zm2 = hm1 @ Wm2 + self.bm2
        hm2 = softplus(zm2)
        outm = hm2 @ Wmo

        # Branch A (unconstrained tanh)
        za1 = xa @ self.Wa1 + self.ba1
        ha1 = tanh_act(za1)
        za2 = ha1 @ self.Wa2 + self.ba2
        ha2 = tanh_act(za2)
        outa = ha2 @ self.Wao + self.bao

        out = outc + outmc + outm + outa + self.bias

        self.cache = dict(
            xc=xc, xmc=xmc, xm=xm, xa=xa,
            Wc1=Wc1, zc1=zc1, hc1=hc1, Wc2=Wc2, zc2=zc2, hc2=hc2, Wco=Wco,
            Wmc1=Wmc1, zmc1=zmc1, hmc1=hmc1, Wmc2=Wmc2, zmc2=zmc2, hmc2=hmc2, Wmco=Wmco,
            Wm1=Wm1, zm1=zm1, hm1=hm1, Wm2=Wm2, zm2=zm2, hm2=hm2, Wmo=Wmo,
            za1=za1, ha1=ha1, za2=za2, ha2=ha2,
        )
        return out

    def backward(self, X, y, out):
        n = X.shape[0]
        c = self.cache
        dout = 2.0 * (out - y) / n  # dMSE/dout, shape (n,1)

        grads = []

        # Branch C backward
        # out_c = hc2 @ Wco
        dWco = c['hc2'].T @ dout             # (h,1)
        dhc2 = dout @ softplus(self.Wcor).T  # (n,h); derivative flows through Wco (positive)
        # dhc2 chain: hc2=softplus(zc2), dzc2 = dhc2 * sp_grad(zc2)
        dzc2 = dhc2 * softplus_grad(c['zc2'])
        # zc2 = hc1 @ Wc2 + bc2
        dWc2 = c['hc1'].T @ dzc2            # (h,h)
        dbc2 = dzc2.sum(axis=0, keepdims=True)
        dhc1 = dzc2 @ softplus(self.Wc2r).T
        dzc1 = dhc1 * softplus_grad(c['zc1'])
        dWc1 = c['xc'].T @ dzc1
        dbc1 = dzc1.sum(axis=0, keepdims=True)
        # Chain through softplus(raw): d_raw = d_pos * sp_grad(raw)
        dWcor_raw = dWco * softplus_grad(self.Wcor)
        dWc2r_raw = dWc2 * softplus_grad(self.Wc2r)
        dWc1r_raw = dWc1 * softplus_grad(self.Wc1r)
        grads += [dWc1r_raw, dbc1, dWc2r_raw, dbc2, dWcor_raw]

        # Branch MC backward (same structure)
        dWmco = c['hmc2'].T @ dout
        dhmc2 = dout @ softplus(self.Wmcor).T
        dzmc2 = dhmc2 * softplus_grad(c['zmc2'])
        dWmc2 = c['hmc1'].T @ dzmc2
        dbmc2 = dzmc2.sum(axis=0, keepdims=True)
        dhmc1 = dzmc2 @ softplus(self.Wmc2r).T
        dzmc1 = dhmc1 * softplus_grad(c['zmc1'])
        dWmc1 = c['xmc'].T @ dzmc1
        dbmc1 = dzmc1.sum(axis=0, keepdims=True)
        dWmcor_raw = dWmco * softplus_grad(self.Wmcor)
        dWmc2r_raw = dWmc2 * softplus_grad(self.Wmc2r)
        dWmc1r_raw = dWmc1 * softplus_grad(self.Wmc1r)
        grads += [dWmc1r_raw, dbmc1, dWmc2r_raw, dbmc2, dWmcor_raw]

        # Branch M backward
        dWmo = c['hm2'].T @ dout
        dhm2 = dout @ softplus(self.Wmor).T
        dzm2 = dhm2 * softplus_grad(c['zm2'])
        dWm2 = c['hm1'].T @ dzm2
        dbm2 = dzm2.sum(axis=0, keepdims=True)
        dhm1 = dzm2 @ softplus(self.Wm2r).T
        dzm1 = dhm1 * softplus_grad(c['zm1'])
        dWm1 = c['xm'].T @ dzm1
        dbm1 = dzm1.sum(axis=0, keepdims=True)
        dWmor_raw = dWmo * softplus_grad(self.Wmor)
        dWm2r_raw = dWm2 * softplus_grad(self.Wm2r)
        dWm1r_raw = dWm1 * softplus_grad(self.Wm1r)
        grads += [dWm1r_raw, dbm1, dWm2r_raw, dbm2, dWmor_raw]

        # Branch A backward (unconstrained tanh)
        dWao = c['ha2'].T @ dout
        dbao = dout.sum(axis=0, keepdims=True)
        dha2 = dout @ self.Wao.T
        dza2 = dha2 * tanh_grad(c['za2'])
        dWa2 = c['ha1'].T @ dza2
        dba2 = dza2.sum(axis=0, keepdims=True)
        dha1 = dza2 @ self.Wa2.T
        dza1 = dha1 * tanh_grad(c['za1'])
        dWa1 = c['xa'].T @ dza1
        dba1 = dza1.sum(axis=0, keepdims=True)
        grads += [dWa1, dba1, dWa2, dba2, dWao, dbao]

        dbias = dout.sum(axis=0, keepdims=True)
        grads += [dbias]

        return grads

    def adam_update(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        params = self._get_params()
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g**2
            m_hat = self.m[i] / (1 - beta1**self.t)
            v_hat = self.v[i] / (1 - beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, X):
        return self.forward(X)


class ISNN2_NumPy:
    """
    ISNN-2: Two convex branches, manual backprop with NumPy.
    """
    def __init__(self, hidden=16, lr=1e-3):
        self.lr = lr
        h = hidden
        scale = 0.1

        # Branch C1
        self.Wc1r = np.random.randn(1, h) * scale
        self.bc1  = np.zeros((1, h))
        self.Wc2r = np.random.randn(h, h) * scale
        self.bc2  = np.zeros((1, h))
        self.Wcor = np.random.randn(h, 1) * scale

        # Branch C2
        self.Wc21r = np.random.randn(1, h) * scale
        self.bc21  = np.zeros((1, h))
        self.Wc22r = np.random.randn(h, h) * scale
        self.bc22  = np.zeros((1, h))
        self.Wc2or = np.random.randn(h, 1) * scale

        self.bias = np.zeros((1, 1))

        self.t = 0
        self.params = self._get_params()
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def _get_params(self):
        return [self.Wc1r, self.bc1, self.Wc2r, self.bc2, self.Wcor,
                self.Wc21r, self.bc21, self.Wc22r, self.bc22, self.Wc2or,
                self.bias]

    def forward(self, X):
        xc1 = X[:, 0:1]
        xc2 = X[:, 1:2]

        Wc1 = softplus(self.Wc1r); Wc2 = softplus(self.Wc2r); Wco = softplus(self.Wcor)
        zc1 = xc1 @ Wc1 + self.bc1
        hc1 = softplus(zc1)
        zc2 = hc1 @ Wc2 + self.bc2
        hc2 = softplus(zc2)
        out1 = hc2 @ Wco

        Wc21 = softplus(self.Wc21r); Wc22 = softplus(self.Wc22r); Wc2o = softplus(self.Wc2or)
        zc21 = xc2 @ Wc21 + self.bc21
        hc21 = softplus(zc21)
        zc22 = hc21 @ Wc22 + self.bc22
        hc22 = softplus(zc22)
        out2 = hc22 @ Wc2o

        out = out1 + out2 + self.bias
        self.cache = dict(
            xc1=xc1, xc2=xc2,
            Wc1=Wc1, zc1=zc1, hc1=hc1, Wc2=Wc2, zc2=zc2, hc2=hc2, Wco=Wco,
            Wc21=Wc21, zc21=zc21, hc21=hc21, Wc22=Wc22, zc22=zc22, hc22=hc22, Wc2o=Wc2o,
        )
        return out

    def backward(self, X, y, out):
        n = X.shape[0]
        c = self.cache
        dout = 2.0 * (out - y) / n

        grads = []

        # Branch C1
        dWco = c['hc2'].T @ dout
        dhc2 = dout @ softplus(self.Wcor).T
        dzc2 = dhc2 * softplus_grad(c['zc2'])
        dWc2 = c['hc1'].T @ dzc2
        dbc2 = dzc2.sum(axis=0, keepdims=True)
        dhc1 = dzc2 @ softplus(self.Wc2r).T
        dzc1 = dhc1 * softplus_grad(c['zc1'])
        dWc1 = c['xc1'].T @ dzc1
        dbc1 = dzc1.sum(axis=0, keepdims=True)
        grads += [dWc1 * softplus_grad(self.Wc1r), dbc1,
                  dWc2 * softplus_grad(self.Wc2r), dbc2,
                  dWco * softplus_grad(self.Wcor)]

        # Branch C2
        dWc2o = c['hc22'].T @ dout
        dhc22 = dout @ softplus(self.Wc2or).T
        dzc22 = dhc22 * softplus_grad(c['zc22'])
        dWc22 = c['hc21'].T @ dzc22
        dbc22 = dzc22.sum(axis=0, keepdims=True)
        dhc21 = dzc22 @ softplus(self.Wc22r).T
        dzc21 = dhc21 * softplus_grad(c['zc21'])
        dWc21 = c['xc2'].T @ dzc21
        dbc21 = dzc21.sum(axis=0, keepdims=True)
        grads += [dWc21 * softplus_grad(self.Wc21r), dbc21,
                  dWc22 * softplus_grad(self.Wc22r), dbc22,
                  dWc2o * softplus_grad(self.Wc2or)]

        dbias = dout.sum(axis=0, keepdims=True)
        grads += [dbias]

        return grads

    def adam_update(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        params = self._get_params()
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g**2
            m_hat = self.m[i] / (1 - beta1**self.t)
            v_hat = self.v[i] / (1 - beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, X):
        return self.forward(X)


def train_numpy(model, X_tr, y_tr, X_te, y_te, epochs=3000):
    train_losses, test_losses = [], []
    for ep in range(epochs):
        out = model.forward(X_tr)
        mse_tr = np.mean((out - y_tr)**2)
        grads = model.backward(X_tr, y_tr, out)
        model.adam_update(grads)

        out_te = model.predict(X_te)
        mse_te = np.mean((out_te - y_te)**2)

        train_losses.append(mse_tr)
        test_losses.append(mse_te)

        if (ep+1) % 500 == 0:
            print(f"  Epoch {ep+1:4d} | Train MSE: {mse_tr:.6f} | Test MSE: {mse_te:.6f}")

    return train_losses, test_losses


np.random.seed(42)
print("\n=== NumPy: ISNN-1 on Toy Problem 1 ===")
model1_np = ISNN1_NumPy(hidden=16, lr=1e-3)
tr1_np, te1_np = train_numpy(model1_np, X1_train, y1_train, X1_test, y1_test, epochs=3000)

np.random.seed(42)
print("\n=== NumPy: ISNN-2 on Toy Problem 2 ===")
model2_np = ISNN2_NumPy(hidden=16, lr=1e-3)
tr2_np, te2_np = train_numpy(model2_np, X2_train, y2_train, X2_test, y2_test, epochs=3000)


# ============================================================
# PLOTTING
# ============================================================
epochs_range = np.arange(1, 3001)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ISNN Training & Testing Loss (Fig 3 & Fig 5 style)", fontsize=14, fontweight='bold')

ax = axes[0, 0]
ax.semilogy(epochs_range, tr1_pt, label='Train Loss', color='blue')
ax.semilogy(epochs_range, te1_pt, label='Test Loss', color='orange', linestyle='--')
ax.set_title("ISNN-1 (PyTorch) - Toy Problem 1")
ax.set_xlabel("Epochs"); ax.set_ylabel("MSE Loss (log scale)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.semilogy(epochs_range, tr1_np, label='Train Loss', color='blue')
ax.semilogy(epochs_range, te1_np, label='Test Loss', color='orange', linestyle='--')
ax.set_title("ISNN-1 (NumPy Manual) - Toy Problem 1")
ax.set_xlabel("Epochs"); ax.set_ylabel("MSE Loss (log scale)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.semilogy(epochs_range, tr2_pt, label='Train Loss', color='green')
ax.semilogy(epochs_range, te2_pt, label='Test Loss', color='red', linestyle='--')
ax.set_title("ISNN-2 (PyTorch) - Toy Problem 2")
ax.set_xlabel("Epochs"); ax.set_ylabel("MSE Loss (log scale)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.semilogy(epochs_range, tr2_np, label='Train Loss', color='green')
ax.semilogy(epochs_range, te2_np, label='Test Loss', color='red', linestyle='--')
ax.set_title("ISNN-2 (NumPy Manual) - Toy Problem 2")
ax.set_xlabel("Epochs"); ax.set_ylabel("MSE Loss (log scale)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved: loss_curves.png")


# ============================================================
# BEHAVIORAL RESPONSE PLOTS (Fig 4 & Fig 6 style)
# ============================================================

# Fig 4: ISNN-1 behavioral response - vary one input at a time, fix others at midpoint
mid = 2.0
x_range = np.linspace(0, 4, 200)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("ISNN-1 Behavioral Response (Fig 4 style)", fontsize=13, fontweight='bold')

input_labels = ['x (convex)', 'y (mono+convex)', 'z (monotone)', 't (arbitrary)']
true_funcs = [
    lambda v: v**2 + np.exp(mid) + mid + np.sin(mid),
    lambda v: mid**2 + np.exp(v) + mid + np.sin(mid),
    lambda v: mid**2 + np.exp(mid) + v + np.sin(mid),
    lambda v: mid**2 + np.exp(mid) + mid + np.sin(v),
]

for col, (label, true_fn) in enumerate(zip(input_labels, true_funcs)):
    for row, (m, name) in enumerate([(model1_pt, 'PyTorch'), (model1_np, 'NumPy')]):
        ax = axes[row, col]
        X_sweep = np.full((200, 4), mid)
        X_sweep[:, col] = x_range

        if name == 'PyTorch':
            m.eval()
            with torch.no_grad():
                pred = m(torch.tensor(X_sweep, dtype=torch.float32)).numpy().flatten()
        else:
            pred = m.predict(X_sweep).flatten()

        true_vals = true_fn(x_range)
        ax.plot(x_range, true_vals, 'k-', linewidth=2, label='True')
        ax.plot(x_range, pred, 'b--', linewidth=2, label=f'ISNN-1 ({name})')
        ax.set_title(f'{name}: vary {label}')
        ax.set_xlabel(f'Input value'); ax.set_ylabel('Output')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('behavioral_isnn1.png', dpi=150, bbox_inches='tight')
print("Saved: behavioral_isnn1.png")


# Fig 6: ISNN-2 behavioral response - 2D surface plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("ISNN-2 Behavioral Response (Fig 6 style)", fontsize=13, fontweight='bold')

x_vals = np.linspace(0, 4, 50)
y_vals = np.linspace(0, 4, 50)
XX, YY = np.meshgrid(x_vals, y_vals)
X_grid = np.column_stack([XX.ravel(), YY.ravel()])
ZZ_true = (XX**2 * YY**2)

for row, (m, name) in enumerate([(model2_pt, 'PyTorch'), (model2_np, 'NumPy')]):
    if name == 'PyTorch':
        m.eval()
        with torch.no_grad():
            ZZ_pred = m(torch.tensor(X_grid, dtype=torch.float32)).numpy().reshape(50, 50)
    else:
        ZZ_pred = m.predict(X_grid).reshape(50, 50)

    ax = axes[row, 0]
    cf = ax.contourf(XX, YY, ZZ_true, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'True: x²·y²')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axes[row, 1]
    cf = ax.contourf(XX, YY, ZZ_pred, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'ISNN-2 ({name}) Prediction')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axes[row, 2]
    err = np.abs(ZZ_pred - ZZ_true)
    cf = ax.contourf(XX, YY, err, levels=20, cmap='Reds')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'Abs Error ({name})')
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
plt.savefig('behavioral_isnn2.png', dpi=150, bbox_inches='tight')
print("Saved: behavioral_isnn2.png")

print("\n=== Final Test MSE Summary ===")
print(f"ISNN-1 PyTorch  | Test MSE: {te1_pt[-1]:.6f}")
print(f"ISNN-1 NumPy    | Test MSE: {te1_np[-1]:.6f}")
print(f"ISNN-2 PyTorch  | Test MSE: {te2_pt[-1]:.6f}")
print(f"ISNN-2 NumPy    | Test MSE: {te2_np[-1]:.6f}")
print("\nDone! All outputs generated.")