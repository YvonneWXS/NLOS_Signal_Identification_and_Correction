# -*- coding: utf-8 -*-
"""
run_urbannav.py -- HK UrbanNav Module 1 Training & Evaluation
=============================================================
Reuses model/loss/training from GAT_V2025.py.
"""
import os, sys, pickle, json, math, time
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
    """Dataset from HK pre-processed graph dicts."""
    def __init__(self, data_list):
        self.data = []
        for d in data_list:
            nf, ei, nl, pe = d["node_features"], d["edge_index"], d["nlos_labels"], d["pseudorange_errors"]
            if d.get("num_satellites", len(nl)) == 0:
                continue
            ea = np.ones(ei.shape[1], dtype=np.float32) * 0.5
            self.data.append({"node_features": nf.astype(np.float32), "edge_index": ei.astype(np.int64),
                              "edge_attr": ea, "pseudorange_error": pe.astype(np.float32),
                              "nlos_label": nl.astype(np.float32)})
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return (torch.tensor(d["node_features"]), torch.tensor(d["edge_index"]),
                torch.tensor(d["edge_attr"]), torch.tensor(d["pseudorange_error"]),
                torch.tensor(d["nlos_label"]))

def load_hk_data():
    base = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data\processedData\UrbanNav-HK_TST\processed"
    with open(os.path.join(base, "train_dataset.pkl"), "rb") as f: train = pickle.load(f)
    with open(os.path.join(base, "val_dataset.pkl"), "rb") as f: val = pickle.load(f)
    return train, val

def compute_stats(ds):
    lbs = np.concatenate([d["nlos_label"] for d in ds.data])
    return {"epochs": len(ds), "total_sats": len(lbs), "nlos_count": int(lbs.sum()),
            "nlos_pct": float(100*lbs.mean()), "avg_sats": float(np.mean([len(d["nlos_label"]) for d in ds.data]))}

def run_urbannav_training():
    config = Config()
    config.NUM_EPOCHS = 100
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 5e-5
    config.GRADIENT_ACCUMULATION = 1
    config.USE_MIXTURE_GAUSSIAN = True
    config.USE_BLOCK_DIAGONAL = True
    config.USE_AMP = True
    config.MOG_PURE_BCE_EPOCHS = 8
    config.MOG_BLEND_EPOCHS = 25
    config.EARLY_STOPPING_PATIENCE = 60
    config.AUTO_POS_WEIGHT = True
    config.LAMBDA_ENTROPY = 0.01
    config.SIGMA_NLOS_CLAMP_LOG_MAX = 3.0

    exp_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result\exp_hk"
    os.makedirs(exp_dir, exist_ok=True)
    tb_dir = os.path.join(exp_dir, "tensorboard"); os.makedirs(tb_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "="*60 + "\nLoading HK UrbanNav Data\n" + "="*60)
    train_raw, val_raw = load_hk_data()
    train_ds = UrbanNavDataset(train_raw); val_ds = UrbanNavDataset(val_raw)
    ts = compute_stats(train_ds); vs = compute_stats(val_ds)
    nlos_ratio = (ts["nlos_count"]+vs["nlos_count"])/(ts["total_sats"]+vs["total_sats"])
    print(f"Train: {ts['epochs']} ep, {ts['total_sats']} sats, NLOS={ts['nlos_pct']:.1f}%")
    print(f"Val:   {vs['epochs']} ep, {vs['total_sats']} sats, NLOS={vs['nlos_pct']:.1f}%")

    if config.AUTO_POS_WEIGHT:
        if nlos_ratio < 0.30: config.POS_WEIGHT = min(2.0, 0.5/max(nlos_ratio,0.01))
        else: config.POS_WEIGHT = (1.0-nlos_ratio)/max(nlos_ratio,0.01)
        print(f"Auto POS_WEIGHT: {config.POS_WEIGHT:.3f}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True,
                              collate_fn=G.batch_collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE*2, shuffle=False,
                            num_workers=config.VAL_NUM_WORKERS, pin_memory=True,
                            collate_fn=G.batch_collate_fn, drop_last=False)

    model = G.NLOSGAT(
        in_features=config.IN_FEATURES, hidden_features=config.HIDDEN_FEATURES,
        num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Loss functions -- exact signatures from GAT_V2025.py main()
    nlos_loss_bce = G.NLOSLoss(
        pos_weight=config.POS_WEIGHT, label_smoothing=config.LABEL_SMOOTHING,
        lambda_bce=config.LAMBDA_BCE, p_los_smoothing=config.P_LOS_SMOOTHING,
        lambda_entropy=config.LAMBDA_ENTROPY, lambda_unc=config.LAMBDA_UNC,
        lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR,
    )
    bce_only_loss = G.NLOSLoss(
        pos_weight=config.POS_WEIGHT, label_smoothing=config.LABEL_SMOOTHING,
        lambda_bce=config.LAMBDA_BCE, p_los_smoothing=config.P_LOS_SMOOTHING,
        lambda_entropy=config.LAMBDA_ENTROPY, lambda_unc=0.0,
        lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR,
    ) if config.USE_MIXTURE_GAUSSIAN else None
    loss_fn = nlos_loss_bce
    mog_loss_fn = G.MoGNLLLoss(
        lambda_entropy=config.LAMBDA_ENTROPY,
        lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR,
        lambda_mu_reg=config.LAMBDA_MU_REG,
        lambda_sigma_reg=config.LAMBDA_SIGMA_REG,
        sigma_gap_target=config.SIGMA_GAP_TARGET,
        lambda_sigma_sep=config.LAMBDA_SIGMA_SEP,
        mu_target=config.MU_NLOS_TARGET,
    ) if config.USE_MIXTURE_GAUSSIAN else None
    mu_reg_loss_fn = G.SupervisedMuRegressionLoss().to(device)
    mu_direction_loss_fn = G.MuDirectionLoss(
        ordering_margin=config.MU_DIRECTION_MARGIN,
    ).to(device)
    sup_loss_fn = G.SupervisedComponentNLLLoss(
        lambda_mu_reg=config.LAMBDA_MU_REG, lambda_sigma_reg=config.LAMBDA_SIGMA_REG,
        sigma_gap_target=config.SIGMA_GAP_TARGET, lambda_sigma_sep=config.LAMBDA_SIGMA_SEP,
        mu_target=config.MU_NLOS_TARGET,
    ) if config.USE_MIXTURE_GAUSSIAN else None

    optimizer, scheduler = G.create_optimizer_and_scheduler(model, config)
    scaler = GradScaler("cuda") if config.USE_AMP and device.type=="cuda" else None
    print(f"AMP: {'enabled' if scaler else 'disabled'}")
    writer = SummaryWriter(log_dir=tb_dir)

    print(f"\n{'='*60}")
    print(f"Training HK UrbanNav (MoG, {config.NUM_EPOCHS} epochs)")
    print(f"BCE warmup: 1-{config.MOG_PURE_BCE_EPOCHS}, Blend: {config.MOG_PURE_BCE_EPOCHS+1}-{config.MOG_PURE_BCE_EPOCHS+config.MOG_BLEND_EPOCHS}")
    print(f"{'='*60}")

    best_val_f1, best_epoch = 0.0, 0
    history = {"train_loss":[], "val_loss":[], "val_acc":[], "val_f1":[]}
    start = time.time()

    for epoch in range(config.NUM_EPOCHS):
        train_metrics = G.train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            epoch, config.GRADIENT_CLIP, config.GRADIENT_ACCUMULATION,
            config.LOG_INTERVAL, writer, epoch*len(train_loader),
            scaler, config.USE_AMP,
            mog_loss_fn=mog_loss_fn, nlos_loss_bce=nlos_loss_bce,
            mog_pure_bce_epochs=config.MOG_PURE_BCE_EPOCHS,
            mog_blend_epochs=config.MOG_BLEND_EPOCHS,
            sup_loss_fn=sup_loss_fn, bce_only_loss=bce_only_loss,
            mu_reg_loss_fn=mu_reg_loss_fn, mu_direction_loss_fn=mu_direction_loss_fn,
        )
        val_metrics = G.evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_metrics.get("loss",float("nan")))
        history["val_loss"].append(val_metrics.get("loss",float("nan")))
        history["val_acc"].append(val_metrics.get("accuracy",0))
        history["val_f1"].append(val_metrics.get("f1",0))

        elapsed = (time.time()-start)/60
        print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
              f"T Loss={train_metrics.get('loss',0):.4f} | "
              f"V Loss={val_metrics.get('loss',0):.4f} | "
              f"Acc={val_metrics.get('accuracy',0):.4f} F1={val_metrics.get('f1',0):.4f} | {elapsed:.1f}min")

        if writer: writer.add_scalar("Epoch/Val_F1", val_metrics.get("f1",0), epoch)

        if val_metrics.get("f1",0) > best_val_f1:
            best_val_f1 = val_metrics["f1"]; best_epoch = epoch+1
            G._safe_torch_save({"epoch":epoch, "model_state_dict":model.state_dict(),
                                "optimizer_state_dict":optimizer.state_dict(), "val_metrics":val_metrics},
                               os.path.join(exp_dir,"best_model.pth"))

        if (epoch+1)%10==0:
            G._safe_torch_save({"epoch":epoch, "model_state_dict":model.state_dict(),
                                "optimizer_state_dict":optimizer.state_dict(), "history":history},
                               os.path.join(exp_dir,f"checkpoint_epoch_{epoch+1:03d}.pth"))

        if config.EARLY_STOPPING_PATIENCE and (epoch+1-best_epoch)>config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}"); break

    total_t = (time.time()-start)/60
    print(f"\nDone: {total_t:.1f}min, Best F1={best_val_f1:.4f} @ ep {best_epoch}")
    G._safe_torch_save({"epoch":config.NUM_EPOCHS, "model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(), "history":history},
                       os.path.join(exp_dir,"final_model.pth"))
    if writer: writer.close()

    # --- Analysis ---
    ckpt = torch.load(os.path.join(exp_dir,"best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    ap, al, am, asn, asl, ae, ac = [],[],[],[],[],[],[]
    with torch.no_grad():
        for nf,ei,ea,pe,lb in val_loader:
            if nf.size(0)==0: continue
            nf,ei = nf.to(device),ei.to(device)
            p,mu,sl,sn = model(nf,ei)
            ap.append(p.cpu().numpy().flatten()); al.append(lb.cpu().numpy().flatten())
            am.append(mu.cpu().numpy().flatten()); asn.append(torch.exp(sn).cpu().numpy().flatten())
            asl.append(torch.exp(sl).cpu().numpy().flatten())
            ae.append(nf[:,0].cpu().numpy()*90.0); ac.append(nf[:,2].cpu().numpy()*60.0)
    pl=np.concatenate(ap); lb=np.concatenate(al); mu=np.concatenate(am)
    sn=np.concatenate(asn); slo=np.concatenate(asl); el=np.concatenate(ae); cn=np.concatenate(ac)

    los_m=lb==0; nlos_m=lb==1
    pred=(pl<0.5).astype(int)
    tp=((pred==1)&(lb==1)).sum(); fp=((pred==1)&(lb==0)).sum()
    tn=((pred==0)&(lb==0)).sum(); fn=((pred==0)&(lb==1)).sum()
    acc=(tp+tn)/len(lb); prec=tp/(tp+fp) if(tp+fp)>0 else 0
    rec=tp/(tp+fn) if(tp+fn)>0 else 0; f1=2*prec*rec/(prec+rec) if(prec+rec)>0 else 0
    p_gap=pl[los_m].mean()-pl[nlos_m].mean() if nlos_m.any() else 0
    sg_gap=sn[nlos_m].mean()-sn[los_m].mean() if nlos_m.any() else 0
    fn_m=(pred==0)&(lb==1); fp_m=(pred==1)&(lb==0)

    report = f"""# UrbanNav-HK_TST Module 1 Analysis Report

## Experiment
- **Dataset**: UrbanNav-HK_TST (Hong Kong TST, 2021-05-17)
- **Model**: NLOSGAT MoG, {n_params:,} params
- **Training**: {total_t:.1f}min, {config.NUM_EPOCHS} epochs, batch={config.BATCH_SIZE} (block-diag), AMP
- **Train**: {ts['epochs']} ep, {ts['total_sats']} sats, NLOS={ts['nlos_pct']:.1f}%
- **Val**: {vs['epochs']} ep, {vs['total_sats']} sats, NLOS={vs['nlos_pct']:.1f}%

## Classification
| Metric | Value |
|--------|:-----:|
| Accuracy | {acc:.4f} |
| Precision | {prec:.4f} |
| Recall | {rec:.4f} |
| F1 | {f1:.4f} |
| Best epoch | {best_epoch} |
| TP/FP/TN/FN | {tp}/{fp}/{tn}/{fn} |

## p_los Distribution
| Group | Mean |
|-------|:----:|
| LOS ({los_m.sum()}) | {pl[los_m].mean():.4f} |
| NLOS ({nlos_m.sum()}) | {pl[nlos_m].mean():.4f} |
| **Gap** | **{p_gap:.4f}** |

## Uncertainty
| Metric | LOS | NLOS | Gap |
|--------|:---:|:----:|:---:|
| sigma_nlos (km) | {sn[los_m].mean():.4f} | {sn[nlos_m].mean():.4f} | {sg_gap:.4f} |
| sigma_los (km) | {slo[los_m].mean():.4f} | {slo[nlos_m].mean():.4f} | — |
| mu_nlos (km) | {mu[los_m].mean():.4f} | {mu[nlos_m].mean():.4f} | {mu[nlos_m].mean()-mu[los_m].mean():.4f} |

## Error Cases
- **FN ({fn})**: elev={el[fn_m].mean():.1f}deg, C/N0={cn[fn_m].mean():.1f}, p_los={pl[fn_m].mean():.4f}
- **FP ({fp})**: elev={el[fp_m].mean():.1f}deg, C/N0={cn[fp_m].mean():.1f}, p_los={pl[fp_m].mean():.4f}

## Cross-Geography Comparison
| Dataset | NLOS% | F1 | p_los Gap | sigma Gap |
|---------|:-----:|:--:|:---------:|:---------:|
| berlin1 | 48.3% | 0.87 | ~0.65 | — |
| berlin2 | 38.8% | 0.87 | ~0.65 | — |
| **HK-TST** | **{nlos_ratio*100:.1f}%** | **{f1:.4f}** | **{p_gap:.4f}** | **{sg_gap:.4f}** |

*Note: Direct F1 comparison limited by different NLOS ratios. Focus on p_los gap generalization.*
"""
    with open(os.path.join(exp_dir,"result.md"),"w",encoding="utf-8") as f: f.write(report)

    results = {"p_los":pl.tolist(),"labels":lb.tolist(),"mu_nlos":mu.tolist(),
               "sigma_nlos":sn.tolist(),"elevation":el.tolist(),"cno":cn.tolist(),
               "metrics":{"accuracy":float(acc),"f1":float(f1),"precision":float(prec),
                          "recall":float(rec),"p_los_gap":float(p_gap),"sigma_gap":float(sg_gap),
                          "best_epoch":best_epoch,"best_val_f1":float(best_val_f1)},
               "history":{k:[float(x) if x==x else None for x in v] for k,v in history.items()}}
    with open(os.path.join(exp_dir,"predictions.json"),"w") as f: json.dump(results,f,indent=2)

    env_md = f"""# Env
- Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
- NLOSGAT MoG ({n_params:,} params), epochs={config.NUM_EPOCHS}, batch={config.BATCH_SIZE} (block-diag)
- LR={config.LEARNING_RATE}, AMP={config.USE_AMP}
- MoG: BCE={config.MOG_PURE_BCE_EPOCHS}ep, Blend={config.MOG_BLEND_EPOCHS}ep
- HK-TST: {ts['epochs']+vs['epochs']} ep, {nlos_ratio*100:.1f}% NLOS, WUM MGEX SP3
"""
    with open(os.path.join(exp_dir,"env.md"),"w",encoding="utf-8") as f: f.write(env_md)

    print(f"\nResults -> {exp_dir} | Acc={acc:.4f} F1={f1:.4f} p_gap={p_gap:.4f} sg_gap={sg_gap:.4f}")
    return results

if __name__=="__main__":
    run_urbannav_training()
