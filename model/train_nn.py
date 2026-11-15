"""EarthScope AI: red neuronal que predice el estado de la Tierra mes a mes.

Entrada: los últimos HIST meses de las variables NASA (anomalías normalizadas).
Salida: el mes siguiente. Para el largo plazo se encadena (rollout autorregresivo).

Fase 1: entrena con 2001-2020, valida con 2021-2022 (parada temprana), evalúa con 2023-2024.
Fase 2 (--final): reentrena con todo hasta 2024 las mismas épocas y predice desde ahí.
Con --seeds N entrena un conjunto (ensemble): la predicción es la media y la dispersión es la incertidumbre.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--repo", default=str(ROOT))
p.add_argument("--out", default=str(ROOT / "build" / "nn"))
p.add_argument("--hist", type=int, default=12)
p.add_argument("--epochs", type=int, default=80, help="máximo; corta antes con --patience")
p.add_argument("--patience", type=int, default=10)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--width", type=int, default=48)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--unroll", type=int, default=2, help="pasos encadenados en la loss (estabilidad a largo plazo)")
p.add_argument("--no-aug", action="store_true", help="sin giro aleatorio en longitud")
p.add_argument("--seeds", type=int, default=1, help="modelos del ensemble")
p.add_argument("--final", action="store_true", help="reentrena con todo hasta 2024 antes de predecir")
p.add_argument("--rollout", type=int, default=72, help="meses a predecir después de 2024-12")
p.add_argument("--coarsen", type=int, default=1, help="2 = 1°, 4 = 2° (pruebas rápidas)")
p.add_argument("--max-steps", type=int, default=0, help="corta cada época (solo pruebas)")
args = p.parse_args()
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, args.repo)
from science.dataset import grid, load_cube  # noqa: E402
from science.variables import GLOBAL_VARIABLES  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
amp = dev.type == "cuda"

# datos
VARS = list(GLOBAL_VARIABLES)
cubes, months = [], None
for v in VARS:
    c, months = load_cube(v)
    c = np.asarray(c, dtype=np.float32)
    if args.coarsen > 1:
        k = args.coarsen
        T, H, W = c.shape
        c = np.nanmean(c.reshape(T, H // k, k, W // k, k), axis=(2, 4))
    cubes.append(c)
X = np.stack(cubes, axis=1)  # (T, V, H, W)
del cubes
T, V, H, W = X.shape
years = np.array([int(m[:4]) for m in months])
moy = np.array([int(m[5:7]) - 1 for m in months])
print(f"datos: {T} meses, {V} variables, {H}x{W}, {months[0]}..{months[-1]}, device={dev}")

# climatología y escala, solo con años de entrenamiento
tr = years <= 2020
clim = np.stack([np.nanmean(X[tr & (moy == m)], axis=0) for m in range(12)])  # (12, V, H, W)
anom = X - clim[moy]
std = np.array([np.nanstd(anom[tr][:, i]) for i in range(V)], dtype=np.float32)
valid = np.isfinite(anom)
Z = np.where(valid, anom / std[None, :, None, None], 0).astype(np.float32)
del anom, X
land = valid[:, VARS.index("evi")].any(0) if "evi" in VARS else np.ones((H, W), bool)

lat, _ = grid()
if args.coarsen > 1:
    lat = lat.reshape(-1, args.coarsen).mean(1)
lat_r = np.deg2rad(lat)[:, None].repeat(W, 1)
STATIC = np.stack([np.sin(lat_r), np.cos(lat_r), land.astype(np.float32)]).astype(np.float32)
np.savez(out / "norm.npz", clim=clim, std=std, vars=np.array(VARS), land=land)

Zt = torch.from_numpy(Z)
Mt = torch.from_numpy(valid)
St = torch.from_numpy(STATIC).to(dev)
IN_CH = args.hist * V + STATIC.shape[0] + 3


def targets(y0: int, y1: int) -> list[int]:
    # índices de meses de salida entre y0 e y1, con espacio para el unroll dentro del rango
    return [t for t in range(args.hist, T - args.unroll + 1)
            if y0 <= years[t] and years[t + args.unroll - 1] <= y1]


def make_input(hist: torch.Tensor, ts: list[int], static: torch.Tensor | None = None) -> torch.Tensor:
    # hist (B, hist, V, H, W) en dev -> canales: historia + estáticos + mes y año de la salida
    B = hist.shape[0]
    ex = []
    for t in ts:
        mm, yy = t % 12, 2001 + t // 12
        a = 2 * math.pi * mm / 12
        ex.append([math.sin(a), math.cos(a), (yy + mm / 12 - 2012) / 12])
    ex = torch.tensor(ex, dtype=hist.dtype, device=hist.device)[:, :, None, None].expand(B, 3, H, W)
    static = St if static is None else static
    return torch.cat([hist.reshape(B, -1, H, W), static.to(hist.dtype).expand(B, -1, H, W), ex], 1)


# modelo: U-Net con borde circular en longitud (el globo da la vuelta)
class Conv(nn.Module):
    def __init__(self, a, b, drop):
        super().__init__()
        self.c = nn.Conv2d(a, b, 3)
        self.n = nn.GroupNorm(8, b)
        self.d = nn.Dropout2d(drop)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")
        return self.d(F.gelu(self.n(self.c(x))))


class Block(nn.Module):
    def __init__(self, a, b, drop):
        super().__init__()
        self.a, self.b = Conv(a, b, drop), Conv(b, b, drop)
        self.skip = nn.Conv2d(a, b, 1) if a != b else nn.Identity()

    def forward(self, x):
        return self.b(self.a(x)) + self.skip(x)


class UNet(nn.Module):
    def __init__(self, cin, cout, w, drop):
        super().__init__()
        ch = [w, w * 2, w * 4, w * 8]
        self.inp = Block(cin, ch[0], drop)
        self.down = nn.ModuleList([Block(ch[i], ch[i + 1], drop) for i in range(3)])
        self.up = nn.ModuleList([Block(ch[i + 1] + ch[i], ch[i], drop) for i in reversed(range(3))])
        self.head = nn.Conv2d(ch[0], cout, 1)
        # arranca prediciendo anomalía 0, es decir la climatología
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = [self.inp(x)]
        for d in self.down:
            skips.append(d(F.avg_pool2d(skips[-1], 2, ceil_mode=True)))
        y = skips.pop()
        for u in self.up:
            s = skips.pop()
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            y = u(torch.cat([y, s], 1))
        return self.head(y)


def masked_mse(pred, y, m):
    m = m.float()
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1)


def step_loss(model, ts: list[int], aug: bool) -> torch.Tensor:
    # pérdida encadenada: predice t, lo usa como entrada para t+1, ...
    idx = torch.tensor(ts)
    hist = torch.stack([Zt[t - args.hist:t] for t in ts]).to(dev, non_blocking=True)
    ys = torch.stack([Zt[t:t + args.unroll] for t in ts]).to(dev)   # (B, U, V, H, W)
    ms = torch.stack([Mt[t:t + args.unroll] for t in ts]).to(dev)
    # giro aleatorio en longitud: el mismo clima visto desde otro meridiano
    shift = int(torch.randint(W, (1,))) if aug else 0
    hist, ys, ms, static = (a.roll(shift, -1) for a in (hist, ys, ms, St))
    total = 0.0
    for k in range(args.unroll):
        x = make_input(hist, (idx + k).tolist(), static)
        with torch.autocast(dev.type, enabled=amp):
            pr = model(x).float()
        total = total + masked_mse(pr, ys[:, k], ms[:, k]) / args.unroll
        hist = torch.cat([hist[:, 1:], pr[:, None]], 1)
    return total


@torch.no_grad()
def predict(models, ts: list[int]) -> torch.Tensor:
    # media del ensemble para un paso
    hist = torch.stack([Zt[t - args.hist:t] for t in ts]).to(dev)
    x = make_input(hist, ts)
    preds = []
    for mdl in models:
        with torch.autocast(dev.type, enabled=amp):
            preds.append(mdl(x).float())
    return torch.stack(preds).mean(0), hist[:, -1]


@torch.no_grad()
def evaluate(models, ts: list[int]):
    for mdl in models:
        mdl.eval()
    se = np.zeros(V); se_pers = np.zeros(V); se_clim = np.zeros(V); n = np.zeros(V)
    for i in range(0, len(ts), args.batch):
        b = ts[i:i + args.batch]
        pr, last = predict(models, b)
        y, m = Zt[b].to(dev), Mt[b].to(dev).float()
        se += (((pr - y) ** 2) * m).sum((0, 2, 3)).cpu().numpy()
        se_pers += (((last - y) ** 2) * m).sum((0, 2, 3)).cpu().numpy()
        se_clim += ((y ** 2) * m).sum((0, 2, 3)).cpu().numpy()
        n += m.sum((0, 2, 3)).cpu().numpy()
    f = lambda s: np.sqrt(s / np.maximum(n, 1)) * std  # noqa: E731
    return f(se), f(se_pers), f(se_clim)


def train(seed: int, train_ts: list[int], val_ts: list[int] | None, epochs: int, tag: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UNet(IN_CH, V, args.width, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    steps = math.ceil(len(train_ts) / args.batch)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=epochs * steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=amp)
    best, best_ep, wait, log = float("inf"), epochs, 0, []
    ckpt = out / f"{tag}.pt"
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(train_ts)
        t0, tot = time.time(), 0.0
        for s in range(steps):
            loss = step_loss(model, list(perm[s * args.batch:(s + 1) * args.batch]), not args.no_aug)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        line = f"[{tag}] época {ep + 1:3d}  loss {tot / steps:.4f}"
        if val_ts:
            rm, _, rc = evaluate([model], val_ts)
            score = float(np.mean(rm / rc))  # < 1 = mejor que la climatología
            log.append({"epoch": ep + 1, "train_loss": tot / steps, "val_skill": score})
            line += f"  val (modelo/clima) {score:.3f}"
            if score < best:
                best, best_ep, wait = score, ep + 1, 0
                torch.save(model.state_dict(), ckpt)
            else:
                wait += 1
        print(f"{line}  {time.time() - t0:.0f}s", flush=True)
        if val_ts and wait >= args.patience:
            print(f"[{tag}] parada temprana, mejor época {best_ep}")
            break
    if val_ts:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
    else:
        torch.save(model.state_dict(), ckpt)
    model.eval()
    return model, best_ep, log


# fase 1: validación y test
tr_ts, va_ts, te_ts = targets(2001, 2020), targets(2021, 2022), targets(2023, 2024)
models, best_eps, logs = [], [], {}
for s in range(args.seeds):
    m, be, lg = train(s, tr_ts, va_ts, args.epochs, f"modelo_{s}")
    models.append(m); best_eps.append(be); logs[f"modelo_{s}"] = lg

rm, rp, rc = evaluate(models, te_ts)
report = {VARS[i]: {"units": GLOBAL_VARIABLES[VARS[i]]["units"], "rmse_modelo": float(rm[i]),
                    "rmse_persistencia": float(rp[i]), "rmse_climatologia": float(rc[i]),
                    "mejora_vs_climatologia_pct": float(100 * (1 - rm[i] / rc[i]))} for i in range(V)}
print(f"\ntest 2023-2024, ensemble de {len(models)} (RMSE, menor es mejor)")
for k, r in report.items():
    print(f"  {k:15s} modelo {r['rmse_modelo']:.3f}  persistencia {r['rmse_persistencia']:.3f}  "
          f"climatología {r['rmse_climatologia']:.3f}  {r['units']}  ({r['mejora_vs_climatologia_pct']:+.1f}%)")

# fase 2: reentrenar con todo
if args.final:
    ep_final = max(1, int(np.median(best_eps)))
    all_ts = targets(2001, 2024)
    print(f"\nfase final: {len(all_ts)} meses hasta 2024, {ep_final} épocas por modelo")
    models = [train(100 + s, all_ts, None, ep_final, f"final_{s}")[0] for s in range(args.seeds)]

(out / "report.json").write_text(json.dumps(
    {"test": report, "args": vars(args), "best_epochs": best_eps, "final": args.final, "log": logs}, indent=1))

# predicción a futuro: media y dispersión del ensemble
land_only = np.array([GLOBAL_VARIABLES[v].get("land_only", False) for v in VARS])
hist = Zt[T - args.hist:].to(dev)[None].repeat(len(models), 1, 1, 1, 1)  # cada modelo su propia trayectoria
means, spreads, labels = [], [], []
with torch.no_grad():
    for k in range(args.rollout):
        t = T + k
        mm, yy = t % 12, 2001 + t // 12
        zs = []
        for j, mdl in enumerate(models):
            x = make_input(hist[j:j + 1], [t])
            with torch.autocast(dev.type, enabled=amp):
                zs.append(mdl(x).float()[0])
        z = torch.stack(zs)                      # (S, V, H, W)
        hist = torch.cat([hist[:, 1:], z[:, None]], 1)
        phys = z.cpu().numpy() * std[None, :, None, None] + clim[mm][None]
        mean, spread = phys.mean(0), phys.std(0)
        for arr in (mean, spread):
            arr[:, ~land] = np.where(land_only[:, None], np.nan, arr[:, ~land])
        means.append(mean.astype(np.float32)); spreads.append(spread.astype(np.float32))
        labels.append(f"{yy}-{mm + 1:02d}")
np.savez_compressed(out / "forecast.npz", data=np.stack(means), spread=np.stack(spreads),
                    months=np.array(labels), vars=np.array(VARS))
print(f"\npredicción {labels[0]}..{labels[-1]} en {out / 'forecast.npz'}")
