# -*- coding: utf-8 -*-
"""
run_urbannav.py -- HK UrbanNav Dataset Module 1 Training & Evaluation
=====================================================================
Reuses model/loss/training from GAT_V2025.py with HK pre-processed data.
"""
import os, sys, pickle, json, math, time, io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
import GAT_V2025 as G

class UrbanNavDataset(Dataset):
    """Dataset wrapping HK pre-processed graph dicts."""
    def __init__(self, data_list):
        self.data = []
        for d in data_list:
            nf = d["node_features"]
            ei = d["edge_index"]
            nl = d["nlos_labels"]
            pe = d["pseudorange_errors"]
            N = d["num_satellites"]
            if N == 0:
                continue
            ea = np.ones(ei.shape[1], dtype=np.float32) * 0.5
            self.data.append({
                "node_features": nf.astype(np.float32),
                "edge_index": ei.astype(np.int64),
                "edge_attr": ea,
                "pseudorange_error": pe.astype(np.float32),
                "nlos_label": nl.astype(np.float32),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            torch.tensor(d["node_features"], dtype=torch.float32),
            torch.tensor(d["edge_index"], dtype=torch.long),
            torch.tensor(d["edge_attr"], dtype=torch.float32),
            torch.tensor(d["pseudorange_error"], dtype=torch.float32),
            torch.tensor(d["nlos_label"], dtype=torch.float32),
        )

def load_hk_data():
    base = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data\processedData\UrbanNav-HK_TST\processed"
    with open(os.path.join(base, "train_dataset.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(base, "val_dataset.pkl"), "rb") as f:
        val_data = pickle.load(f)
    return train_data, val_data

def compute_stats(dataset):
    all_labels = []
    for d in dataset.data:
        all_labels.extend(d["nlos_label"].tolist())
    all_labels = np.array(all_labels)
    return {
        "epochs": len(dataset),
        "total_sats": len(all_labels),
        "nlos_count": int(all_labels.sum()),
        "nlos_pct": float(100 * all_labels.mean()),
        "avg_sats": float(np.mean([len(dd["nlos_label"]) for dd in dataset.data])),
    }

def run_urbannav_training():
    config = Config()
    config.USE_MIXTURE_GAUSSIAN = True
    config.NUM_EPOCHS = 100
    config.BATCH_SIZE = 32
    config.USE_BLOCK_DIAGONAL = True
    config.USE_AMP = True
    config.LEARNING_RATE = 5e-5
    config.GRADIENT_ACCUMULATION = 1
    config.LOG_INTERVAL = 10
    config.USE_TENSORBOARD = True
    config.MOG_PURE_BCE_EPOCHS = 8
    config.MOG_BLEND_EPOCHS = 25
    config.EARLY_STOPPING_PATIENCE = 60
    config.AUTO_POS_WEIGHT = True
    # HK-specific: lower entropy reg for low-NLOS scenario
    config.LAMBDA_ENTROPY = 0.01
    config.SIGMA_NLOS_CLAMP_LOG_MAX = 3.0
    config.LAMBDA_MU_REG = 0.20

    exp_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result\exp_hk"
    os.makedirs(exp_dir, exist_ok=True)
    tb_dir = os.path.join(exp_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("\n" + "="*60)
    print("Loading HK UrbanNav Data")
    print("="*60)
    train_raw, val_raw = load_hk_data()
    train_dataset = UrbanNavDataset(train_raw)
    val_dataset = UrbanNavDataset(val_raw)
    tr_s = compute_stats(train_dataset)
    vl_s = compute_stats(val_dataset)
    nlos_ratio = (tr_s["nlos_count"] + vl_s["nlos_count"]) / (tr_s["total_sats"] + vl_s["total_sats"])
    print(f"Train: {tr_s['epochs']} ep, {tr_s['total_sats']} sats, NLOS={tr_s['nlos_pct']:.1f}%")
    print(f"Val:   {vl_s['epochs']} ep, {vl_s['total_sats']} sats, NLOS={vl_s['nlos_pct']:.1f}%")

    if config.AUTO_POS_WEIGHT:
        if nlos_ratio < 0.30:
            config.POS_WEIGHT = min(2.0, 0.5 / max(nlos_ratio, 0.01))
        else:
            los_ratio = 1.0 - nlos_ratio
            config.POS_WEIGHT = los_ratio / max(nlos_ratio, 0.01)
        print(f"Auto POS_WEIGHT: {config.POS_WEIGHT:.3f} (NLOS={nlos_ratio*100:.1f}%)")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True, collate_fn=G.batch_collate_fn, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 2,
                            shuffle=False, num_workers=config.VAL_NUM_WORKERS,
                            pin_memory=True, collate_fn=G.batch_collate_fn, drop_last=False)

    # Model
    model = G.NLOSGAT(
        in_features=config.IN_FEATURES, hidden_features=config.HIDDEN_FEATURES,
        num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        sigma_los_clamp_log_min=config.SIGMA_LOS_CLAMP_LOG_MIN,
        sigma_los_clamp_log_max=config.SIGMA_LOS_CLAMP_LOG_MAX,
        sigma_nlos_clamp_log_min=config.SIGMA_NLOS_CLAMP_LOG_MIN,
        sigma_nlos_clamp_log_max=config.SIGMA_NLOS_CLAMP_LOG_MAX,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Loss functions (using GAT_V2025.py classes)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.POS_WEIGHT])).to(device)
    mog_loss_fn = G.NLOSLoss(
        lambda_entropy=config.LAMBDA_ENTROPY,
        lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR,
    ).to(device)
    sup_loss_fn = G.SupervisedMuRegressionLoss(
        lambda_mu_reg=config.LAMBDA_MU_REG,
        lambda_mu_warmup_reg=config.LAMBDA_MU_WARMUP_REG,
    ).to(device)
    mu_direction_loss_fn = G.MuDirectionLoss(
        lambda_mu_direction=config.LAMBDA_MU_DIRECTION,
        los_target=config.MU_DIRECTION_LOS_TARGET,
        margin=config.MU_DIRECTION_MARGIN,
    ).to(device)

    optimizer, scheduler = G.create_optimizer_and_scheduler(model, config)
    scaler = GradScaler("cuda") if config.USE_AMP and device.type == "cuda" else None
    print(f"AMP: {'enabled' if scaler else 'disabled'}")
    writer = SummaryWriter(log_dir=tb_dir)

    print(f"\n{'='*60}")
    print(f"Training HK UrbanNav (MoG, {config.NUM_EPOCHS} epochs)")
    print(f"BCE warmup: 1-{config.MOG_PURE_BCE_EPOCHS}, Blend: {config.MOG_PURE_BCE_EPOCHS+1}-{config.MOG_PURE_BCE_EPOCHS+config.MOG_BLEND_EPOCHS}, NLL: {config.MOG_PURE_BCE_EPOCHS+config.MOG_BLEND_EPOCHS+1}-{config.NUM_EPOCHS}")
    print(f"{'='*60}")

    best_val_f1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        train_metrics = G.train_epoch(
            model, train_loader, optimizer, scheduler, bce_loss_fn, device,
            epoch, config.GRADIENT_CLIP, config.GRADIENT_ACCUMULATION,
            config.LOG_INTERVAL, writer, epoch * len(train_loader),
            scaler, config.USE_AMP,
            mog_loss_fn, mog_loss_fn,
            config.MOG_PURE_BCE_EPOCHS, config.MOG_BLEND_EPOCHS,
            sup_loss_fn, bce_loss_fn, None, mu_direction_loss_fn
        )

        val_metrics = G.evaluate(model, val_loader, bce_loss_fn, device)

        history["train_loss"].append(train_metrics.get("loss", float("nan")))
        history["val_loss"].append(val_metrics.get("loss", float("nan")))
        history["val_acc"].append(val_metrics.get("accuracy", 0))
        history["val_f1"].append(val_metrics.get("f1", 0))

        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
              f"Train Loss={train_metrics.get('loss',0):.4f} | "
              f"Val Loss={val_metrics.get('loss',0):.4f} | "
              f"Acc={val_metrics.get('accuracy',0):.4f} F1={val_metrics.get('f1',0):.4f} | "
              f"{elapsed:.1f}min")

        if writer:
            writer.add_scalar("Epoch/Val_F1", val_metrics.get("f1", 0), epoch)

        if val_metrics.get("f1", 0) > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            G._safe_torch_save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, os.path.join(exp_dir, "best_model.pth"))

        if (epoch + 1) % 10 == 0:
            G._safe_torch_save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "history": history,
            }, os.path.join(exp_dir, f"checkpoint_epoch_{epoch+1:03d}.pth"))

        if config.EARLY_STOPPING_PATIENCE and (epoch + 1 - best_epoch) > config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    total_time = (time.time() - start_time) / 60
    print(f"\nDone: {total_time:.1f} min, Best Val F1={best_val_f1:.4f} @ epoch {best_epoch}")

    G._safe_torch_save({
        "epoch": config.NUM_EPOCHS, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "history": history,
    }, os.path.join(exp_dir, "final_model.pth"))
    if writer: writer.close()

    # --- Analysis ---
    ckpt = torch.load(os.path.join(exp_dir, "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_p_los, all_labels, all_mu, all_snlos, all_slos, all_elev, all_cno = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            nf, ei, ea, pe, lb = batch
            if nf.size(0) == 0: continue
            nf, ei = nf.to(device), ei.to(device)
            p, mu, sl, sn = model(nf, ei)
            all_p_los.append(p.cpu().numpy().flatten())
            all_labels.append(lb.cpu().numpy().flatten())
            all_mu.append(mu.cpu().numpy().flatten())
            all_snlos.append(torch.exp(sn).cpu().numpy().flatten())
            all_slos.append(torch.exp(sl).cpu().numpy().flatten())
            all_elev.append(nf[:, 0].cpu().numpy() * 90.0)
            all_cno.append(nf[:, 2].cpu().numpy() * 60.0)

    pl = np.concatenate(all_p_los); lb = np.concatenate(all_labels)
    mu = np.concatenate(all_mu); sn = np.concatenate(all_snlos)
    slo = np.concatenate(all_slos); el = np.concatenate(all_elev)
    cn = np.concatenate(all_cno)

    los_m = lb == 0; nlos_m = lb == 1
    pred = (pl < 0.5).astype(int)
    tp = ((pred==1)&(lb==1)).sum(); fp = ((pred==1)&(lb==0)).sum()
    tn = ((pred==0)&(lb==0)).sum(); fn = ((pred==0)&(lb==1)).sum()
    acc = (tp+tn)/len(lb); prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0; f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

    fn_m = (pred==0)&(lb==1); fp_m = (pred==1)&(lb==0)
    p_gap = pl[los_m].mean() - pl[nlos_m].mean() if nlos_m.any() else 0
    sg_gap = sn[nlos_m].mean() - sn[los_m].mean() if nlos_m.any() and los_m.any() else 0

    report = f"""# UrbanNav-HK_TST Module 1 Analysis Report

## Experiment Summary
- **Dataset**: UrbanNav-HK_TST (Hong Kong TST, May 17 2021)
- **Train**: {tr_s['epochs']} ep, {tr_s['total_sats']} sats, NLOS={tr_s['nlos_pct']:.1f}%
- **Val**: {vl_s['epochs']} ep, {vl_s['total_sats']} sats, NLOS={vl_s['nlos_pct']:.1f}%
- **Model**: NLOSGAT (MoG), {n_params:,} params
- **Training**: {total_time:.1f} min, {config.NUM_EPOCHS} epochs

## Classification Performance
| Metric | Value |
|--------|:-----:|
| Accuracy | {acc:.4f} |
| Precision | {prec:.4f} |
| Recall | {rec:.4f} |
| F1 Score | {f1:.4f} |
| Best epoch | {best_epoch} |
| TP/FP/TN/FN | {tp}/{fp}/{tn}/{fn} |

## p_los Distribution
| Group | Mean p_los |
|-------|:----------:|
| LOS ({los_m.sum()}) | {pl[los_m].mean():.4f} |
| NLOS ({nlos_m.sum()}) | {pl[nlos_m].mean():.4f} |
| **Gap** | **{p_gap:.4f}** |

## Uncertainty (sigma_nlos) Analysis
| Group | Mean sigma_nlos (km) | Mean sigma_los (km) |
|-------|:-------------------:|:-------------------:|
| LOS | {sn[los_m].mean():.4f} | {slo[los_m].mean():.4f} |
| NLOS | {sn[nlos_m].mean():.4f} | {slo[nlos_m].mean():.4f} |
| Gap | {sg_gap:.4f} | — |

## mu_NLOS Analysis
| Group | Mean mu_NLOS (km) |
|-------|:-----------------:|
| LOS | {mu[los_m].mean():.4f} |
| NLOS | {mu[nlos_m].mean():.4f} |

## Error Cases
### FN ({fn}): NLOS predicted as LOS
- Elevation: {el[fn_m].mean():.1f} deg | C/N0: {cn[fn_m].mean():.1f} dBHz
- p_los: {pl[fn_m].mean():.4f} | sigma_nlos: {sn[fn_m].mean():.4f} km

### FP ({fp}): LOS predicted as NLOS
- Elevation: {el[fp_m].mean():.1f} deg | C/N0: {cn[fp_m].mean():.1f} dBHz
- p_los: {pl[fp_m].mean():.4f} | sigma_nlos: {sn[fp_m].mean():.4f} km

## Elevation Breakdown
| Elevation | Count | NLOS% | p_los(LOS) | p_los(NLOS) |
|-----------|:-----:|:-----:|:----------:|:-----------:|
"""
    for lo,hi in [(0,15),(15,30),(30,45),(45,60),(60,90)]:
        m = (el>=lo)&(el<hi)
        if m.sum()==0: continue
        lm, nm = m&los_m, m&nlos_m
        report += f"| {lo}-{hi} deg | {m.sum()} | {nm.sum()/max(m.sum(),1)*100:.1f}% | {pl[lm].mean() if lm.any() else 0:.4f} | {pl[nm].mean() if nm.any() else 0:.4f} |\n"

    report += f"""
## Key Findings
1. HK low-NLOS scenario ({nlos_ratio*100:.1f}% NLOS) tests cross-geography generalization
2. p_los gap: {p_gap:.4f} (target > 0.3) -> {"GOOD separation" if p_gap > 0.3 else "MODERATE — low NLOS limits separation"}
3. sigma_nlos gap: {sg_gap:.4f} km -> {"GOOD uncertainty differentiation" if sg_gap > 0.3 else "LIMITED — low NLOS hampers sigma learning"}
4. mu_nlos(NLOS) = {mu[nlos_m].mean():.4f} km — {"reasonable" if 0.05 < mu[nlos_m].mean() < 0.5 else "check calibration"}

## Comparison: European vs Asian Urban Canyons
| Dataset | NLOS% | F1 | p_los Gap | sigma_nlos Gap |
|---------|:-----:|:--:|:---------:|:-------------:|
| berlin1 (Potsdamer Platz) | 48.3% | ~0.87 | ~0.65 | — |
| berlin2 (Gendarmenmarkt) | 38.8% | ~0.87 | ~0.65 | — |
| **HK-TST** | **{nlos_ratio*100:.1f}%** | **{f1:.4f}** | **{p_gap:.4f}** | **{sg_gap:.4f}** |

*Note: Direct F1 comparison is unfair due to vastly different NLOS ratios. Focus on p_los gap generalization.*
"""
    with open(os.path.join(exp_dir, "result.md"), "w", encoding="utf-8") as f:
        f.write(report)

    # Save predictions
    results = {
        "p_los": pl.tolist(), "labels": lb.tolist(), "mu_nlos": mu.tolist(),
        "sigma_nlos": sn.tolist(), "sigma_los": slo.tolist(),
        "elevation": el.tolist(), "cno": cn.tolist(),
        "metrics": {"accuracy": float(acc), "precision": float(prec),
                    "recall": float(rec), "f1": float(f1),
                    "p_los_gap": float(p_gap), "sigma_nlos_gap": float(sg_gap),
                    "best_epoch": best_epoch, "best_val_f1": float(best_val_f1)},
        "dataset_stats": tr_s, "history": {k: [float(x) if x==x else None for x in v] for k,v in history.items()},
    }
    with open(os.path.join(exp_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)

    env_md = f"""# Experiment Environment
- **Python**: {sys.version.split()[0]}
- **PyTorch**: {torch.__version__}
- **CUDA**: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
- **GPU**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
- **Model**: NLOSGAT MoG, {n_params:,} params
- **Epochs**: {config.NUM_EPOCHS}, Batch={config.BATCH_SIZE} (block-diag)
- **LR**: {config.LEARNING_RATE}, AMP={config.USE_AMP}
- **MoG schedule**: BCE={config.MOG_PURE_BCE_EPOCHS}, Blend={config.MOG_BLEND_EPOCHS}
- **Dataset**: UrbanNav-HK_TST, {tr_s['epochs']+vl_s['epochs']} epochs, {nlos_ratio*100:.1f}% NLOS
"""
    with open(os.path.join(exp_dir, "env.md"), "w", encoding="utf-8") as f:
        f.write(env_md)

    print(f"\nResults -> {exp_dir}")
    print(f"  Acc={acc:.4f} F1={f1:.4f} p_gap={p_gap:.4f} sn_gap={sg_gap:.4f}")
    return results

if __name__ == "__main__":
    run_urbannav_training()
