"""Stage 2 (FC-02): hand-rolled deep-learning forecasters in pure PyTorch (D-38).

Native sequence-to-sequence, driver-conditional, multi-horizon, partial-pooled. Evaluated
on the SAME observed test points/horizons as FC-01 for direct comparability.

Leak-free windowing:
  - encoder (history, known at origin t): [gap-filled CH4, fx_ drivers, lagged fluxes ar_*_t]
    over a lookback L.
  - decoder (future t+1..t+H): ONLY known-future drivers (fx_ : met/soil/livestock/mgmt/
    calendar). CH4 and FCO2/GPP/Reco are NEVER fed at t+h.
  - static: tower one-hot.
  - target: observed CH4 at t+1..t+H (masked; gap-filled used only for encoder/training-fill).

Models: DLinear (decomposition + linear, the Zeng-2023 simple baseline; +future-exog term),
LSTM seq2seq, LSTM+VSN (variable-selection gate -> native variable importance).

Device-agnostic: uses CUDA only if a real GPU op succeeds (guards the sm_120 case), else CPU.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import full_metrics

HOURLY = Path(__file__).resolve().parents[2] / "data" / "Hourly"
TOWERS = [2, 4, 9]

FX = None   # filled at load (future-exog columns, 19)
FLUX_T = ["ar_fc_t", "ar_gpp_t", "ar_reco_t"]   # realized fluxes at origin (encoder-only)

TRACKS = {
    "A": dict(freq="h", L=168, H=48, heval=[1, 6, 12, 24, 48], stride=6),
    "B": dict(freq="D", L=28,  H=14, heval=[1, 3, 7, 14], stride=1),
}


# ── device ───────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum().item()
            return torch.device("cuda")
        except Exception as e:
            print(f"[forecasting_dl] CUDA present but unusable ({e}); falling back to CPU")
    return torch.device("cpu")


# ── data ─────────────────────────────────────────────────────────────────────
def load_matrix(path=None):
    global FX
    m = pd.read_csv(path or (HOURLY / "forecast_features.csv"), low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], format="mixed")
    FX = [c for c in m.columns if c.startswith("fx")]
    return m


def tower_series(m, t, track):
    """Per-tower aligned arrays for a track: CH4(gap), enc-exog, dec-exog, Y(observed), index."""
    cfg = TRACKS[track]
    df = m[m.tower == t].set_index("Datetime").sort_index()
    if cfg["freq"] == "D":
        cnt = df["y_observed"].resample("D").count()
        yo = df["y_observed"].resample("D").mean().where(cnt >= 6)
        yg = df["y_gapfilled"].resample("D").mean()
        ex = df[FX + FLUX_T].resample("D").mean()
        df = pd.concat([yg.rename("y_gapfilled"), yo.rename("y_observed"), ex], axis=1)
    ch4 = df["y_gapfilled"].to_numpy(np.float32)
    Y = df["y_observed"].to_numpy(np.float32)
    enc_ex = df[FX + FLUX_T].to_numpy(np.float32)         # (N, 19+3)
    dec_ex = df[FX].to_numpy(np.float32)                  # (N, 19)
    return dict(t=t, idx=df.index, ch4=ch4, Y=Y, enc_ex=enc_ex, dec_ex=dec_ex)


def make_windows(ser, L, H, stride=1):
    """Origins (strided) with full lookback + horizon. Returns dict of arrays + origin/target times."""
    ch4, Y, enc_ex, dec_ex, idx = ser["ch4"], ser["Y"], ser["enc_ex"], ser["dec_ex"], ser["idx"]
    N = len(ch4)
    enc_feat = np.concatenate([ch4[:, None], enc_ex], axis=1)   # (N, 1+22)
    enc_feat = np.nan_to_num(enc_feat, nan=0.0)
    dec_ex = np.nan_to_num(dec_ex, nan=0.0)
    oi = np.arange(L - 1, N - H, stride, dtype=int)
    enc = np.stack([enc_feat[i - L + 1:i + 1] for i in oi])     # (n, L, F_enc)
    dec = np.stack([dec_ex[i + 1:i + 1 + H] for i in oi])       # (n, H, F_dec)
    y   = np.stack([Y[i + 1:i + 1 + H] for i in oi])            # (n, H)
    otime = idx[oi]
    ttime = np.stack([idx[i + 1:i + 1 + H].to_numpy() for i in oi])  # (n, H) datetimes
    return dict(enc=enc, dec=dec, y=y, otime=otime, ttime=ttime, t=ser["t"])


def build_windows(m, track):
    """{tower: window_dict} for a track (uses the track's L/H/stride)."""
    cfg = TRACKS[track]
    return {t: make_windows(tower_series(m, t, track), cfg["L"], cfg["H"], cfg["stride"])
            for t in TOWERS}


# ── models ───────────────────────────────────────────────────────────────────
class MovingAvg(nn.Module):
    def __init__(self, k=25):
        super().__init__(); self.k = k; self.pool = nn.AvgPool1d(k, stride=1, padding=0)
    def forward(self, x):   # x: (B, L)
        pad = self.k // 2
        xp = torch.cat([x[:, :1].repeat(1, pad), x, x[:, -1:].repeat(1, pad)], dim=1)
        return self.pool(xp.unsqueeze(1)).squeeze(1)[:, :x.shape[1]]


class DLinear(nn.Module):
    """Decomposition-linear (Zeng 2023) on CH4 lookback + linear future-exog term."""
    def __init__(self, L, H, n_dec, n_static, k=25):
        super().__init__()
        self.ma = MovingAvg(k)
        self.lin_t = nn.Linear(L, H); self.lin_s = nn.Linear(L, H)
        self.lin_x = nn.Linear(H * n_dec + n_static, H)
    def forward(self, enc, dec, static):
        ch4 = enc[:, :, 0]                          # (B, L) — CH4 is feature 0
        trend = self.ma(ch4); seas = ch4 - trend
        base = self.lin_t(trend) + self.lin_s(seas)
        xz = torch.cat([dec.reshape(dec.shape[0], -1), static], dim=1)
        return base + self.lin_x(xz)                # (B, H)


class LSTMSeq2Seq(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, H, hidden=64):
        super().__init__()
        self.enc = nn.LSTM(n_enc, hidden, batch_first=True)
        self.dec = nn.LSTM(n_dec, hidden, batch_first=True)
        self.stat = nn.Linear(n_static, hidden)
        self.out = nn.Linear(hidden, 1); self.H = H
    def forward(self, enc, dec, static):
        _, (h, c) = self.enc(enc)
        h = h + self.stat(static).unsqueeze(0)
        o, _ = self.dec(dec, (h, c))
        return self.out(o).squeeze(-1)              # (B, H)


class LSTMVSN(nn.Module):
    """LSTM with a Variable Selection gate on encoder features -> native importance."""
    def __init__(self, n_enc, n_dec, n_static, H, hidden=64):
        super().__init__()
        self.vsn = nn.Sequential(nn.Linear(n_enc, n_enc), nn.Tanh(), nn.Linear(n_enc, n_enc))
        self.enc = nn.LSTM(n_enc, hidden, batch_first=True)
        self.dec = nn.LSTM(n_dec, hidden, batch_first=True)
        self.stat = nn.Linear(n_static, hidden)
        self.out = nn.Linear(hidden, 1); self.H = H; self.last_w = None
    def forward(self, enc, dec, static):
        w = torch.softmax(self.vsn(enc), dim=-1)    # (B, L, n_enc) variable weights
        self.last_w = w.detach()
        enc = enc * w * enc.shape[-1]               # gated, rescaled
        _, (h, c) = self.enc(enc)
        h = h + self.stat(static).unsqueeze(0)
        o, _ = self.dec(dec, (h, c))
        return self.out(o).squeeze(-1)


# ── TFT (B-03b): canonical Temporal Fusion Transformer (Lim et al. 2021) ──────
# GRN/VSN/static-encoders/interpretable-multi-head-attention, hand-rolled pure PyTorch
# (no pytorch-forecasting, matching the D-38 no-library convention). Modest d_model/n_heads
# (bounded-iteration norm) relative to a research-scale TFT, but architecturally complete.
class GRN(nn.Module):
    """Gated Residual Network: the core TFT building block (skip + ELU-MLP + GLU + LayerNorm)."""
    def __init__(self, d_in, d_hidden, d_out=None, d_context=None, dropout=0.1):
        super().__init__()
        d_out = d_out or d_in
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.context = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.glu = nn.Linear(d_out, d_out * 2)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)
    def forward(self, x, c=None):
        skip = self.skip(x)
        h = self.fc1(x)
        if self.context is not None and c is not None:
            h = h + self.context(c)
        h = self.fc2(torch.nn.functional.elu(h))
        g = self.glu(self.drop(h))
        a, b = g.chunk(2, dim=-1)
        return self.norm(skip + a * torch.sigmoid(b))


class BatchedPerVarGRN(nn.Module):
    """GRN applied independently per variable (separate weights per variable, vectorised via
    einsum rather than a Python loop over variables -- keeps VSN tractable at ~20-30 features."""
    def __init__(self, n_vars, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.has_skip = d_in != d_out
        if self.has_skip:
            self.W_skip = nn.Parameter(torch.randn(n_vars, d_in, d_out) * 0.1)
            self.b_skip = nn.Parameter(torch.zeros(n_vars, d_out))
        self.W1 = nn.Parameter(torch.randn(n_vars, d_in, d_hidden) * 0.1); self.b1 = nn.Parameter(torch.zeros(n_vars, d_hidden))
        self.W2 = nn.Parameter(torch.randn(n_vars, d_hidden, d_out) * 0.1); self.b2 = nn.Parameter(torch.zeros(n_vars, d_out))
        self.Wg = nn.Parameter(torch.randn(n_vars, d_out, d_out * 2) * 0.1); self.bg = nn.Parameter(torch.zeros(n_vars, d_out * 2))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)
    def forward(self, x):   # x: (..., n_vars, d_in)
        skip = torch.einsum("...vi,vio->...vo", x, self.W_skip) + self.b_skip if self.has_skip else x
        h = torch.nn.functional.elu(torch.einsum("...vi,vih->...vh", x, self.W1) + self.b1)
        h = torch.einsum("...vh,vho->...vo", h, self.W2) + self.b2
        g = torch.einsum("...vo,vop->...vp", self.drop(h), self.Wg) + self.bg
        a, b = g.chunk(2, dim=-1)
        return self.norm(skip + a * torch.sigmoid(b))


class VSN(nn.Module):
    """Variable Selection Network: per-variable GRN transform + softmax-gated combination."""
    def __init__(self, n_vars, d_model, d_context=None, dropout=0.1):
        super().__init__()
        self.n_vars, self.d_model = n_vars, d_model
        self.pervar_w = nn.Parameter(torch.randn(n_vars, d_model) * 0.1)
        self.pervar_b = nn.Parameter(torch.zeros(n_vars, d_model))
        self.var_grn = BatchedPerVarGRN(n_vars, d_model, d_model, d_model, dropout=dropout)
        self.weight_grn = GRN(n_vars * d_model, d_model, n_vars, d_context=d_context, dropout=dropout)
    def forward(self, x, c=None):   # x: (B,T,n_vars)
        proj = x.unsqueeze(-1) * self.pervar_w + self.pervar_b      # (B,T,n_vars,d_model)
        flat = proj.reshape(*proj.shape[:-2], self.n_vars * self.d_model)
        weights = torch.softmax(self.weight_grn(flat, c), dim=-1)    # (B,T,n_vars)
        transformed = self.var_grn(proj)                             # (B,T,n_vars,d_model)
        combined = (transformed * weights.unsqueeze(-1)).sum(dim=-2)
        return combined, weights


class GateAddNorm(nn.Module):
    """Gated skip connection (GLU + residual + LayerNorm) -- TFT's locality-enhancement gate."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.glu = nn.Linear(d_model, d_model * 2)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, skip):
        g = self.glu(self.drop(x))
        a, b = g.chunk(2, dim=-1)
        return self.norm(skip + a * torch.sigmoid(b))


class InterpretableMultiHeadAttention(nn.Module):
    """TFT's interpretable multi-head attention: per-head Q/K, a single SHARED value projection
    across heads (unlike vanilla MHA), outputs averaged across heads for interpretability."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads; self.d_head = d_model // n_heads
        self.q_layers = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(n_heads)])
        self.k_layers = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(n_heads)])
        self.v_layer = nn.Linear(d_model, self.d_head)
        self.out = nn.Linear(self.d_head, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        V = self.v_layer(x)
        outs, attns = [], []
        for h in range(self.n_heads):
            Q, K = self.q_layers[h](x), self.k_layers[h](x)
            scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            attn = self.drop(torch.softmax(scores, dim=-1))
            outs.append(attn @ V); attns.append(attn)
        out = torch.stack(outs, dim=0).mean(0)
        return self.out(out), torch.stack(attns, dim=1)


class TFT(nn.Module):
    """Canonical TFT: static covariate encoders -> VSN (encoder/decoder) -> LSTM enc/dec with
    static-initialised state -> gated locality enhancement -> static enrichment -> interpretable
    self-attention (causal) -> position-wise feed-forward -> point-forecast head over the decoder
    horizon. d_model/n_heads kept modest (bounded-iteration norm, D-41) relative to a
    research-scale TFT, but every architectural component from Lim et al. (2021) is present."""
    def __init__(self, L, H, n_enc, n_dec, n_static, d_model=32, n_heads=4, dropout=0.1):
        super().__init__()
        self.L, self.H = L, H
        self.static_vsn = VSN(n_static, d_model, dropout=dropout)
        self.static_ctx_selection = GRN(d_model, d_model, d_model, dropout=dropout)
        self.static_ctx_enrichment = GRN(d_model, d_model, d_model, dropout=dropout)
        self.static_ctx_h = GRN(d_model, d_model, d_model, dropout=dropout)
        self.static_ctx_c = GRN(d_model, d_model, d_model, dropout=dropout)
        self.enc_vsn = VSN(n_enc, d_model, d_context=d_model, dropout=dropout)
        self.dec_vsn = VSN(n_dec, d_model, d_context=d_model, dropout=dropout)
        self.lstm_enc = nn.LSTM(d_model, d_model, batch_first=True)
        self.lstm_dec = nn.LSTM(d_model, d_model, batch_first=True)
        self.gate_lstm = GateAddNorm(d_model, dropout=dropout)
        self.static_enrich_grn = GRN(d_model, d_model, d_model, d_context=d_model, dropout=dropout)
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.gate_attn = GateAddNorm(d_model, dropout=dropout)
        self.pos_ff = GRN(d_model, d_model, d_model, dropout=dropout)
        self.gate_ff = GateAddNorm(d_model, dropout=dropout)
        self.out = nn.Linear(d_model, 1)
        self.last_attn = None
        self.register_buffer("_mask", torch.triu(torch.ones(L + H, L + H, dtype=torch.bool), diagonal=1))
    def forward(self, enc, dec, static):
        static_emb, _ = self.static_vsn(static.unsqueeze(1))
        static_emb = static_emb.squeeze(1)
        c_sel, c_enr = self.static_ctx_selection(static_emb), self.static_ctx_enrichment(static_emb)
        h0, c0 = self.static_ctx_h(static_emb).unsqueeze(0), self.static_ctx_c(static_emb).unsqueeze(0)

        enc_emb, _ = self.enc_vsn(enc, c_sel.unsqueeze(1).expand(-1, enc.shape[1], -1))
        dec_emb, _ = self.dec_vsn(dec, c_sel.unsqueeze(1).expand(-1, dec.shape[1], -1))

        enc_lstm, (h, c) = self.lstm_enc(enc_emb, (h0, c0))
        dec_lstm, _ = self.lstm_dec(dec_emb, (h, c))

        lstm_out = torch.cat([enc_lstm, dec_lstm], dim=1)
        vsn_out = torch.cat([enc_emb, dec_emb], dim=1)
        gated = self.gate_lstm(lstm_out, vsn_out)

        enriched = self.static_enrich_grn(gated, c_enr.unsqueeze(1).expand(-1, gated.shape[1], -1))
        attn_out, attn_w = self.attn(enriched, mask=self._mask)
        self.last_attn = attn_w.detach()
        attn_gated = self.gate_attn(attn_out, enriched)

        ff_out = self.pos_ff(attn_gated)
        final = self.gate_ff(ff_out, gated)
        dec_final = final[:, self.L:, :]
        return self.out(dec_final).squeeze(-1)


def build_model(name, L, H, n_enc, n_dec, n_static):
    if name == "DLinear": return DLinear(L, H, n_dec, n_static)
    if name == "LSTM":    return LSTMSeq2Seq(n_enc, n_dec, n_static, H)
    if name == "LSTM_VSN":return LSTMVSN(n_enc, n_dec, n_static, H)
    if name == "TFT":     return TFT(L, H, n_enc, n_dec, n_static)
    raise ValueError(name)


# ── train / predict ──────────────────────────────────────────────────────────
class Scaler:
    def fit(self, x):   # x: (..., F)
        flat = x.reshape(-1, x.shape[-1]); self.mu = flat.mean(0); self.sd = flat.std(0) + 1e-6; return self
    def tf(self, x): return (x - self.mu) / self.sd

def train_model(model, tr, device, epochs=30, bs=256, lr=1e-3, ch4_mu=0.0, ch4_sd=1.0, seed=0,
                 weight_decay=0.0, val_data=None, patience=5):
    """weight_decay/val_data/patience default to off -- existing callers (DLinear/LSTM/LSTM_VSN in
    B02/B04) are unaffected; AdamW with weight_decay=0 is equivalent to plain Adam. Pass val_data
    (a window dict like `tr`, e.g. a held-out year) to enable early stopping on validation loss --
    added for TFT (B-03b), which overfits the bounded 30-epoch/no-val budget the simpler DL models
    don't (D-45)."""
    import copy
    torch.manual_seed(seed); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    enc = torch.tensor(tr["enc"]); dec = torch.tensor(tr["dec"]); static = torch.tensor(tr["static"])
    y = torch.tensor((tr["y"] - ch4_mu) / ch4_sd); mask = torch.tensor(np.isfinite(tr["y"]).astype(np.float32))
    y = torch.nan_to_num(y, nan=0.0)
    n = len(enc)

    if val_data is not None:
        # keep on CPU, move per mini-batch -- an unbatched forward pass over a large validation set
        # blows up memory/time for attention models (TFT's O(T^2) attention over a batch of thousands
        # allocates a (B,heads,T,T) score tensor that can reach several GB) -- bit us in practice (D-45)
        venc = torch.tensor(val_data["enc"]); vdec = torch.tensor(val_data["dec"]); vstatic = torch.tensor(val_data["static"])
        vy = torch.tensor((val_data["y"] - ch4_mu) / ch4_sd); vmask = torch.tensor(np.isfinite(val_data["y"]).astype(np.float32))
        vy = torch.nan_to_num(vy, nan=0.0)
        nv = len(venc)
        best_val, best_state, bad_epochs = float("inf"), None, 0

    for ep in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            xe, xd, xs = enc[idx].to(device), dec[idx].to(device), static[idx].to(device)
            yb, mb = y[idx].to(device), mask[idx].to(device)
            opt.zero_grad()
            pred = model(xe, xd, xs)
            loss = ((pred - yb) ** 2 * mb).sum() / mb.sum().clamp(min=1)
            loss.backward(); opt.step()
        if val_data is not None:
            model.eval()
            vloss_sum, vmask_sum = 0.0, 0.0
            with torch.no_grad():
                for b in range(0, nv, bs):
                    xe, xd, xs = venc[b:b+bs].to(device), vdec[b:b+bs].to(device), vstatic[b:b+bs].to(device)
                    yb, mb = vy[b:b+bs].to(device), vmask[b:b+bs].to(device)
                    vpred = model(xe, xd, xs)
                    vloss_sum += ((vpred - yb) ** 2 * mb).sum().item(); vmask_sum += mb.sum().item()
            vloss = vloss_sum / max(vmask_sum, 1)
            model.train()
            if vloss < best_val:
                best_val, best_state, bad_epochs = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
    if val_data is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict(model, te, device, ch4_mu, ch4_sd, bs=512):
    model.eval(); outs = []
    enc = torch.tensor(te["enc"]); dec = torch.tensor(te["dec"]); static = torch.tensor(te["static"])
    for b in range(0, len(enc), bs):
        xe, xd, xs = enc[b:b+bs].to(device), dec[b:b+bs].to(device), static[b:b+bs].to(device)
        outs.append(model(xe, xd, xs).cpu().numpy())
    return np.concatenate(outs) * ch4_sd + ch4_mu     # (n, H) in CH4 units


# ── orchestration (CV identical to FC-01) ────────────────────────────────────
TOW = {2: [1., 0., 0.], 4: [0., 1., 0.], 9: [0., 0., 1.]}
T2_FOLDS = [("2018-06-30", "2018-07-01", "2018-12-31"), ("2018-12-31", "2019-01-01", "2019-05-31")]


def _subset(win, mask):
    mask = np.asarray(mask)
    return dict(enc=win["enc"][mask].copy(), dec=win["dec"][mask].copy(), y=win["y"][mask].copy(),
                static=np.tile(TOW[win["t"]], (int(mask.sum()), 1)).astype(np.float32),
                ttime=win["ttime"][mask], otime=win["otime"][mask],
                persist0=win["enc"][mask][:, -1, 0].copy())   # raw CH4 at origin (before scaling)


def _cat(parts):
    out = {}
    for k in ["enc", "dec", "y", "static", "ttime", "otime", "persist0"]:
        out[k] = np.concatenate([p[k] for p in parts]) if parts else None
    return out


def _standardize(train, tests):
    se = Scaler().fit(train["enc"]); sd = Scaler().fit(train["dec"])
    yv = train["y"][np.isfinite(train["y"])]; mu, sdy = float(yv.mean()), float(yv.std() + 1e-6)
    for d in [train] + tests:
        if d["enc"] is not None and len(d["enc"]):
            d["enc"] = se.tf(d["enc"]); d["dec"] = sd.tf(d["dec"])
    return mu, sdy


def _clim(train, unit):
    tt = pd.DatetimeIndex(train["ttime"].reshape(-1)); yy = train["y"].reshape(-1)
    ok = np.isfinite(yy); tt, yy = tt[ok], yy[ok]
    df = pd.DataFrame({"y": yy, "mo": tt.month})
    if unit == "h":
        df["hr"] = tt.hour; g = df.groupby(["mo", "hr"])["y"].mean()
    else:
        g = df.groupby("mo")["y"].mean()
    return g, float(yy.mean())


def _clim_pred(g, gl, ttimes, unit):
    tt = pd.DatetimeIndex(ttimes)
    key = list(zip(tt.month, tt.hour)) if unit == "h" else list(tt.month)
    return np.array([g.get(k, gl) for k in key], float)


def _metrics(y, p):
    from sklearn.metrics import r2_score
    y = np.asarray(y, float); p = np.asarray(p, float)
    r2 = r2_score(y, p) if (len(y) > 1 and np.var(y) > 0) else np.nan
    return float(np.sqrt(np.mean((p - y) ** 2))), float(np.mean(np.abs(p - y))), r2, float(np.mean(p - y))


def _eval_rows(track, model_name, tower, te, preds, gclim, gl, unit):
    cfg = TRACKS[track]; rows = []
    for h in cfg["heval"]:
        k = h - 1
        y = te["y"][:, k]; obs = np.isfinite(y)
        if obs.sum() < 10: continue
        yo = y[obs]
        pd_dl = preds[obs, k]
        pp = te["persist0"][obs]                                   # persistence = origin value
        pc = _clim_pred(gclim, gl, te["ttime"][obs, k], unit)
        r, mae, r2, mbe = _metrics(yo, pd_dl)
        rp, rc = np.sqrt(np.mean((pp - yo) ** 2)), np.sqrt(np.mean((pc - yo) ** 2))
        fm = full_metrics(yo, pd_dl, y_naive=pp)
        rows.append(dict(track=track, horizon=h, tower=tower, model=model_name,
                         RMSE=round(r, 3), MAE=round(mae, 3),
                         R2=(round(r2, 3) if np.isfinite(r2) else np.nan), MBE=round(mbe, 3),
                         n_test=int(obs.sum()),
                         skill_persist=round(1 - r / rp, 3) if rp > 0 else np.nan,
                         skill_clim=round(1 - r / rc, 3) if rc > 0 else np.nan,
                         WAPE=round(fm["WAPE"], 4) if np.isfinite(fm["WAPE"]) else np.nan,
                         MASE=round(fm["MASE"], 4) if np.isfinite(fm["MASE"]) else np.nan,
                         sMAPE=round(fm["sMAPE"], 4) if np.isfinite(fm["sMAPE"]) else np.nan,
                         MAPE=round(fm["MAPE"], 4) if np.isfinite(fm["MAPE"]) else np.nan,
                         MAPE_n_excluded=fm["MAPE_n_excluded"]))
    return rows


def run_track(track, model_name, W, device, epochs=30, seed=0):
    """Full CV for one (track, model): T4/T9 main split + Tower-2 expanding folds."""
    cfg = TRACKS[track]; unit = "h" if cfg["freq"] == "h" else "D"; rows = []
    Wt = W[track]
    n_enc = Wt[4]["enc"].shape[-1]; n_dec = Wt[4]["dec"].shape[-1]
    cut2021 = pd.Timestamp("2021-12-31 23:59")

    # ---- MAIN: towers 4 & 9, train target<=2021 (all towers), test 2022-2023 ----
    tr_parts = [_subset(Wt[t], pd.DatetimeIndex(Wt[t]["ttime"][:, -1]) <= cut2021) for t in [2, 4, 9]]
    train = _cat(tr_parts)
    tests = {t: _subset(Wt[t], pd.DatetimeIndex(Wt[t]["otime"]).year.isin([2022, 2023])) for t in [4, 9]}
    mu, sdy = _standardize(train, list(tests.values()))
    gclim, gl = _clim({"ttime": train["ttime"], "y": train["y"]}, unit)
    model = build_model(model_name, cfg["L"], cfg["H"], n_enc, n_dec, 3)
    train_model(model, train, device, epochs=epochs, ch4_mu=mu, ch4_sd=sdy, seed=seed)
    for t in [4, 9]:
        if len(tests[t]["enc"]) < 10: continue
        preds = predict(model, tests[t], device, mu, sdy)
        rows += _eval_rows(track, model_name, t, tests[t], preds, gclim, gl, unit)

    # ---- Tower 2: expanding-window folds (donor = Tower 4) ----
    acc_pred, acc_te = [], []
    for cutd, s, e in T2_FOLDS:
        cut = pd.Timestamp(cutd + " 23:59")
        trp = [_subset(Wt[t], pd.DatetimeIndex(Wt[t]["ttime"][:, -1]) <= cut) for t in [2, 4, 9]]
        trf = _cat(trp)
        ot2 = pd.DatetimeIndex(Wt[2]["otime"])
        tef = _subset(Wt[2], (ot2 >= pd.Timestamp(s)) & (ot2 <= pd.Timestamp(e)))
        if len(tef["enc"]) < 10 or len(trf["enc"]) < 50: continue
        mu, sdy = _standardize(trf, [tef])
        gclim, gl = _clim({"ttime": trf["ttime"], "y": trf["y"]}, unit)
        model = build_model(model_name, cfg["L"], cfg["H"], n_enc, n_dec, 3)
        train_model(model, trf, device, epochs=epochs, ch4_mu=mu, ch4_sd=sdy, seed=seed)
        acc_pred.append(predict(model, tef, device, mu, sdy)); acc_te.append(tef)
    if acc_te:
        te = {k: np.concatenate([a[k] for a in acc_te]) for k in ["y", "persist0", "ttime"]}
        rows += _eval_rows(track, model_name, 2, te, np.concatenate(acc_pred), gclim, gl, unit)
    return rows


# ── quantile / pinball (FC-03 UQ, D-40) ──────────────────────────────────────
QUANTILES = [0.05, 0.5, 0.95]


class LSTMQuantile(nn.Module):
    """LSTM seq2seq with a quantile head -> (B, H, Q) for pinball-loss UQ."""
    def __init__(self, n_enc, n_dec, n_static, H, nq, hidden=64):
        super().__init__()
        self.enc = nn.LSTM(n_enc, hidden, batch_first=True)
        self.dec = nn.LSTM(n_dec, hidden, batch_first=True)
        self.stat = nn.Linear(n_static, hidden)
        self.out = nn.Linear(hidden, nq); self.H = H
    def forward(self, enc, dec, static):
        _, (h, c) = self.enc(enc); h = h + self.stat(static).unsqueeze(0)
        o, _ = self.dec(dec, (h, c))
        return self.out(o)                          # (B, H, Q)


def pinball_loss(pred, y, mask, quantiles):
    """pred (B,H,Q); y,mask (B,H)."""
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
    e = y.unsqueeze(-1) - pred
    loss = torch.maximum(q * e, (q - 1) * e) * mask.unsqueeze(-1)
    return loss.sum() / mask.sum().clamp(min=1) / len(quantiles)


def train_quantile(model, tr, device, quantiles=QUANTILES, epochs=30, bs=256, lr=1e-3,
                   ch4_mu=0.0, ch4_sd=1.0, seed=0):
    torch.manual_seed(seed); model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    enc = torch.tensor(tr["enc"]); dec = torch.tensor(tr["dec"]); static = torch.tensor(tr["static"])
    y = torch.tensor((tr["y"] - ch4_mu) / ch4_sd); mask = torch.tensor(np.isfinite(tr["y"]).astype(np.float32))
    y = torch.nan_to_num(y, nan=0.0); n = len(enc)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            xe, xd, xs = enc[idx].to(device), dec[idx].to(device), static[idx].to(device)
            yb, mb = y[idx].to(device), mask[idx].to(device)
            opt.zero_grad(); loss = pinball_loss(model(xe, xd, xs), yb, mb, quantiles)
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict_quantile(model, te, device, ch4_mu, ch4_sd, bs=512):
    model.eval(); outs = []
    enc = torch.tensor(te["enc"]); dec = torch.tensor(te["dec"]); static = torch.tensor(te["static"])
    for b in range(0, len(enc), bs):
        outs.append(model(enc[b:b+bs].to(device), dec[b:b+bs].to(device), static[b:b+bs].to(device)).cpu().numpy())
    q = np.concatenate(outs) * ch4_sd + ch4_mu        # (n, H, Q)
    return np.sort(q, axis=-1)                          # enforce non-crossing
