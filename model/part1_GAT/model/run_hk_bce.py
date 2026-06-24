"""HK BCE-only training - reuses GAT_V2025 train/evaluate with USE_MIXTURE_GAUSSIAN=False."""
import os, sys, pickle, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
import GAT_V2025 as G
from run_urbannav import UrbanNavDataset, load_hk_data, compute_stats

config = Config()
config.USE_MIXTURE_GAUSSIAN = False
config.NUM_EPOCHS = 100
config.BATCH_SIZE = 32
config.LEARNING_RATE = 5e-5
config.GRADIENT_ACCUMULATION = 1
config.USE_BLOCK_DIAGONAL = True
config.USE_AMP = True
config.AUTO_POS_WEIGHT = True
config.LAMBDA_ENTROPY = 0.01

exp_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result\exp_hk"
tb_dir = os.path.join(exp_dir, "tensorboard_bce")
os.makedirs(tb_dir, exist_ok=True)

device = torch.device("cuda")
print(f"Device: {device} | {torch.cuda.get_device_name(0)}")

train_raw, val_raw = load_hk_data()
train_ds = UrbanNavDataset(train_raw); val_ds = UrbanNavDataset(val_raw)
ts = compute_stats(train_ds); vs = compute_stats(val_ds)
nlos_ratio = (ts["nlos_count"]+vs["nlos_count"])/(ts["total_sats"]+vs["total_sats"])

# Strong pos_weight for extreme imbalance
config.POS_WEIGHT = min(5.0, 0.3 / max(nlos_ratio, 0.005))
print(f"Train: {ts['epochs']} ep, NLOS={ts['nlos_pct']:.1f}% | Val: NLOS={vs['nlos_pct']:.1f}%")
print(f"POS_WEIGHT: {config.POS_WEIGHT:.3f}")

train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=G.batch_collate_fn, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE*2, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=G.batch_collate_fn, drop_last=False)

model = G.NLOSGAT(in_features=config.IN_FEATURES, hidden_features=config.HIDDEN_FEATURES,
                  num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT).to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

nlos_loss_bce = G.NLOSLoss(pos_weight=config.POS_WEIGHT, label_smoothing=0.0,
                           lambda_bce=config.LAMBDA_BCE, p_los_smoothing=0.0,
                           lambda_entropy=config.LAMBDA_ENTROPY, lambda_unc=config.LAMBDA_UNC,
                           lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR)
loss_fn = nlos_loss_bce

optimizer, scheduler = G.create_optimizer_and_scheduler(model, config)
scaler = GradScaler("cuda") if config.USE_AMP else None
writer = SummaryWriter(log_dir=tb_dir)

best_val_f1, best_epoch = 0.0, 0
t0 = time.time()

for epoch in range(config.NUM_EPOCHS):
    train_metrics = G.train_epoch(
        model, train_loader, optimizer, scheduler, loss_fn, device,
        epoch, config.GRADIENT_CLIP, config.GRADIENT_ACCUMULATION,
        config.LOG_INTERVAL, writer, epoch*len(train_loader),
        scaler, config.USE_AMP,
        mog_loss_fn=None, nlos_loss_bce=None,
        mog_pure_bce_epochs=9999, mog_blend_epochs=0,
        sup_loss_fn=None, bce_only_loss=None,
        mu_reg_loss_fn=None, mu_direction_loss_fn=None,
    )
    val_metrics = G.evaluate(model, val_loader, loss_fn, device)
    
    elapsed = (time.time()-t0)/60
    print(f"Epoch {epoch+1:3d} | T Loss={train_metrics.get('loss',0):.4f} | "
          f"V Acc={val_metrics.get('accuracy',0):.4f} F1={val_metrics.get('f1',0):.4f} | {elapsed:.1f}min")
    
    if writer: writer.add_scalar("Epoch/Val_F1", val_metrics.get("f1",0), epoch)
    
    if val_metrics.get("f1",0) > best_val_f1:
        best_val_f1 = val_metrics["f1"]; best_epoch = epoch+1
        G._safe_torch_save({"epoch":epoch, "model_state_dict":model.state_dict(),
                            "val_metrics":val_metrics}, os.path.join(exp_dir,"best_model_bce.pth"))
    
    if (epoch+1)%10==0:
        G._safe_torch_save({"epoch":epoch, "model_state_dict":model.state_dict()},
                           os.path.join(exp_dir,f"bce_checkpoint_{epoch+1:03d}.pth"))

if writer: writer.close()
print(f"\nDone: {(time.time()-t0)/60:.1f}min, Best F1={best_val_f1:.4f} @ ep {best_epoch}")

# Analysis
ckpt = torch.load(os.path.join(exp_dir,"best_model_bce.pth"), map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
ap, al = [], []
with torch.no_grad():
    for nf,ei,ea,pe,lb in val_loader:
        if nf.size(0)==0: continue
        p,_,_,_ = model(nf.to(device),ei.to(device))
        ap.append(p.cpu().numpy().flatten()); al.append(lb.cpu().numpy().flatten())
pl=np.concatenate(ap); lb=np.concatenate(al)
pred=(pl<0.5).astype(int)
tp=((pred==1)&(lb==1)).sum(); fp=((pred==1)&(lb==0)).sum()
tn=((pred==0)&(lb==0)).sum(); fn=((pred==0)&(lb==1)).sum()
acc=(tp+tn)/len(lb); prec=tp/(tp+fp) if(tp+fp)>0 else 0
rec=tp/(tp+fn) if(tp+fn)>0 else 0; f1=2*prec*rec/(prec+rec) if(prec+rec)>0 else 0
print(f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} | TP={tp} FP={fp} TN={tn} FN={fn}")

results = {"p_los":pl.tolist(),"labels":lb.tolist(),
           "metrics":{"accuracy":float(acc),"f1":float(f1),"precision":float(prec),
                      "recall":float(rec),"plos_gap":float(pl[lb==0].mean()-pl[lb==1].mean()) if lb.sum()>0 else 0},
           "dataset_stats":ts}
with open(os.path.join(exp_dir,"predictions_bce.json"),"w") as f: json.dump(results,f,indent=2)
print("Saved predictions_bce.json")
