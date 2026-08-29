"""
Revision experiment for Reviewer A, comment 3.

The reviewer asks how far the conclusions depend on frequency-based
(population-level) uncertainty quantification, and what happens when the
uncertainty is a full predictive PDF, as produced by Bayesian methods.

We answer with two constructions that return a predictive distribution rather
than a population interval:

  1. MC-dropout MLP -- dropout at test time is an approximate Bayesian
     posterior over the network weights (Gal & Ghahramani, 2016); 100
     stochastic forward passes give a predictive sample.
  2. Gaussian-NLL deep ensemble -- five networks with mean and log-variance
     heads trained by negative log likelihood (Lakshminarayanan et al., 2017);
     the predictive distribution is an equally weighted Gaussian mixture, so it
     carries both aleatoric and epistemic parts.

For each we report, on the in-distribution validation window and on the shifted
late-life test window:
  * coverage of the central 90 % credible interval (comparable with Table 4),
  * CRPS (a proper score for the whole PDF, not just one interval), and
  * the probability integral transform (PIT): its Kolmogorov-Smirnov distance
    from uniform, plus the mass in the tails, which shows in which direction
    the distribution is wrong.

Both models are fit on the training window only, so the validation window is a
genuine out-of-sample reference here.

Output: experiment_results/a3_pdf_uq.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPT_SV", "HPC_SV"]
SEED = 42
N_MC = 100          # MC-dropout forward passes
N_ENS = 5           # deep-ensemble members
EPOCHS = 200
ALPHA = 0.10

torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class DropoutMLP(nn.Module):
    def __init__(self, d_in, p=0.2, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GaussMLP(nn.Module):
    """Mean and log-variance heads -> a Gaussian predictive density."""

    def __init__(self, d_in, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, 1)
        self.logvar = nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.mu(z).squeeze(-1), self.logvar(z).squeeze(-1).clamp(-8, 12)


def train(model, X, y, gaussian, epochs=EPOCHS, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        if gaussian:
            mu, logvar = model(X)
            loss = (0.5 * logvar + 0.5 * (y - mu) ** 2 / logvar.exp()).mean()
        else:
            loss = ((model(X) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return model


def crps_samples(y, samples):
    """Sample-based CRPS: E|X-y| - 0.5 E|X-X'| , X,X' ~ predictive."""
    s = np.sort(samples, axis=1)
    n = s.shape[1]
    term1 = np.mean(np.abs(s - y[:, None]), axis=1)
    # E|X-X'| via the sorted-sample identity
    w = (2 * np.arange(1, n + 1) - n - 1)
    term2 = 2.0 * (s * w).sum(axis=1) / (n * n)
    return float(np.mean(term1 - 0.5 * term2))


def pit_from_samples(y, samples):
    return np.mean(samples <= y[:, None], axis=1)


def ks_uniform(u):
    u = np.sort(u)
    n = len(u)
    e = (np.arange(1, n + 1)) / n
    return float(np.max(np.abs(u - e)))


def summarize(name, target, split, y, samples):
    lo = np.quantile(samples, ALPHA / 2, axis=1)
    hi = np.quantile(samples, 1 - ALPHA / 2, axis=1)
    u = pit_from_samples(y, samples)
    return dict(method=name, target=target, split=split, n=len(y),
                cov90=round(float(((y >= lo) & (y <= hi)).mean()), 3),
                mean_width=round(float(np.mean(hi - lo)), 1),
                crps=round(crps_samples(y, samples), 1),
                pit_ks=round(ks_uniform(u), 3),
                pit_below_05=round(float((u < 0.05).mean()), 3),
                pit_above_95=round(float((u > 0.95).mean()), 3))


def main():
    tr = pd.read_csv(RESULTS / "train_with_predictions.csv")
    va = pd.read_csv(RESULTS / "val_with_predictions.csv")
    te = pd.read_csv(RESULTS / "test_with_predictions.csv")
    drop = ["ESN", "Cycles_Since_New"] + [f"Cycles_to_{t}" for t in TARGETS] \
        + [f"Pred_Cycles_to_{t}" for t in TARGETS]
    feats = [c for c in tr.columns if c not in drop]

    mu_x = tr[feats].mean().to_numpy()
    sd_x = tr[feats].std().replace(0, 1).to_numpy()

    def X(df):
        return torch.tensor(((df[feats].to_numpy() - mu_x) / sd_x),
                            dtype=torch.float32, device=DEV)

    Xtr, Xva, Xte = X(tr), X(va), X(te)
    rows = []
    for t in TARGETS:
        print(f"\n=== {t} ===", flush=True)
        ycol = f"Cycles_to_{t}"
        y_scale = tr[ycol].std()
        ytr = torch.tensor(tr[ycol].to_numpy() / y_scale, dtype=torch.float32,
                           device=DEV)

        # ---------------- MC dropout ----------------
        m = train(DropoutMLP(len(feats)).to(DEV), Xtr, ytr, gaussian=False)
        m.train()   # keep dropout active at prediction time
        with torch.no_grad():
            for split, Xs, df in (("val", Xva, va), ("test", Xte, te)):
                s = np.stack([m(Xs).cpu().numpy() for _ in range(N_MC)], axis=1)
                s = np.clip(s * y_scale, 0, None)
                r = summarize("MC-dropout", t, split, df[ycol].to_numpy(), s)
                rows.append(r)
                print("   ", r, flush=True)

        # ---------------- Gaussian-NLL deep ensemble ----------------
        members = []
        for k in range(N_ENS):
            torch.manual_seed(SEED + k)
            members.append(train(GaussMLP(len(feats)).to(DEV), Xtr, ytr,
                                 gaussian=True))
        with torch.no_grad():
            for split, Xs, df in (("val", Xva, va), ("test", Xte, te)):
                samp = []
                for mm in members:
                    mu, logvar = mm(Xs)
                    mu = mu.cpu().numpy()
                    sd = logvar.exp().sqrt().cpu().numpy()
                    samp.append(mu[:, None] + sd[:, None] *
                                np.random.randn(len(mu), N_MC // N_ENS))
                s = np.clip(np.concatenate(samp, axis=1) * y_scale, 0, None)
                r = summarize("Gaussian-NLL ensemble", t, split,
                              df[ycol].to_numpy(), s)
                rows.append(r)
                print("   ", r, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "a3_pdf_uq.csv", index=False)
    print("\n=== A3: predictive-distribution UQ, validation vs late-life test ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
