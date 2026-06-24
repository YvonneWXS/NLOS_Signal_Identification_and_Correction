import os, sys, json, time, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
import GAT_V2025 as G
from run_urbannav import UrbanNavDataset, load_hk_data, compute_stats

def resume_training(resume_from, target_epochs=100):
    """Resume training from checkpoint."""
    config = Config()
    config.NUM_EPOCHS = target_epochs
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 5e-5
    config.USE_MIXTURE_GAUSSIAN = True
    config.USE_BLOCK_DIAGONAL = True
    config.USE_AMP = True
    config.MOG_PURE_BCE_EPOCHS = 8
    config.MOG_BLEND_EPOCHS = 25
    config.EARLY_STOPPING_PATIENCE = 60
    config.AUTO_POS_WEIGHT = True
    config.LAMBDA_ENTROPY = 0.005  # Lower for very low NLOS
    config.GRADIENT_CLIP = 2.0

    exp_dir = os.path.dirname(resume_from)
    tb_dir = os.path.join(exp_dir, "tensorboard")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_raw, val_raw = load_hk_data()
    train_ds = UrbanNavDataset(train_raw)
    val_ds = UrbanNavDataset(val_raw)
    ts = compute_stats(train_ds)
    nlos_ratio = ts["nlos_count"] / ts["total_sats"]
    print(f"Train: {ts['epochs']} ep, NLOS={ts['nlos_pct']:.1f}%")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=G.batch_collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=G.batch_collate_fn, drop_last=False)

    # Model
    model = G.NLOSGAT(in_features=config.IN_FEATURES, hidden_features=config.HIDDEN_FEATURES,
                      num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT).to(device)

    # Losses (same as main)
    nlos_loss_bce = G.NLOSLoss(pos_weight=config.POS_WEIGHT, label_smoothing=config.LABEL_SMOOTHING,
                               lambda_bce=config.LAMBDA_BCE, p_los_smoothing=config.P_LOS_SMOOTHING,
                               lambda_entropy=config.LAMBDA_ENTROPY, lambda_unc=config.LAMBDA_UNC,
                               lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR)
    bce_only_loss = G.NLOSLoss(pos_weight=config.POS_WEIGHT, label_smoothing=config.LABEL_SMOOTHING,
                               lambda_bce=config.LAMBDA_BCE, p_los_smoothing=config.P_LOS_SMOOTHING,
                               lambda_entropy=config.LAMBDA_ENTROPY, lambda_unc=0.0,
                               lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR)
    loss_fn = nlos_loss_bce
    mog_loss_fn = G.MoGNLLLoss(lambda_entropy=config.LAMBDA_ENTROPY, lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR,
                              lambda_mu_reg=config.LAMBDA_MU_REG, lambda_sigma_reg=config.LAMBDA_SIGMA_REG,
                              sigma_gap_target=config.SIGMA_GAP_TARGET, lambda_sigma_sep=config.LAMBDA_SIGMA_SEP,
                              mu_target=config.MU_NLOS_TARGET)
    mu_reg_loss_fn = G.SupervisedMuRegressionLoss().to(device)
    mu_direction_loss_fn = G.MuDirectionLoss(ordering_margin=config.MU_DIRECTION_MARGIN).to(device)
    sup_loss_fn = G.SupervisedComponentNLLLoss(lambda_mu_reg=config.LAMBDA_MU_REG, lambda_sigma_reg=config.LAMBDA_SIGMA_REG,
                                              sigma_gap_target=config.SIGMA_GAP_TARGET, lambda_sigma_sep=config.LAMBDA_SIGMA_SEP,
                                              mu_target=config.MU_NLOS_TARGET)

    # Load checkpoint
    ckpt = torch.load(resume_from, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}")

    optimizer, scheduler = G.create_optimizer_and_scheduler(model, config)
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except:
            print("  Optimizer state load failed, using fresh optimizer")

    scaler = GradScaler("cuda") if config.USE_AMP else None
    writer = SummaryWriter(log_dir=tb_dir)

    best_val_f1 = ckpt.get("val_metrics", {}).get("f1", 0.0)
    best_epoch = start_epoch - 1
    print(f"Best Val F1 so far: {best_val_f1:.4f}")

    t0 = time.time()
    for epoch in range(start_epoch, target_epochs):
        try:
            train_metrics = G.train_epoch(
                model, train_loader, optimizer, scheduler, loss_fn, device,
                epoch, config.GRADIENT_CLIP, config.GRADIENT_ACCUMULATION,
                config.LOG_INTERVAL, writer, epoch * len(train_loader),
                scaler, config.USE_AMP,
                mog_loss_fn=mog_loss_fn, nlos_loss_bce=nlos_loss_bce,
                mog_pure_bce_epochs=config.MOG_PURE_BCE_EPOCHS,
                mog_blend_epochs=config.MOG_BLEND_EPOCHS,
                sup_loss_fn=sup_loss_fn, bce_only_loss=bce_only_loss,
                mu_reg_loss_fn=mu_reg_loss_fn, mu_direction_loss_fn=mu_direction_loss_fn,
            )
            val_metrics = G.evaluate(model, val_loader, loss_fn, device)
        except Exception as e:
            print(f"  ERROR at epoch {epoch+1}: {e}")
            continue

        elapsed = (time.time() - t0) / 60
        print(f"Epoch {epoch+1:3d}/{target_epochs} | "
              f"T Loss={train_metrics.get('loss',0):.4f} | "
              f"V Loss={val_metrics.get('loss',0):.4f} | "
              f"Acc={val_metrics.get('accuracy',0):.4f} F1={val_metrics.get('f1',0):.4f} | {elapsed:.1f}min")

        if val_metrics.get("f1", 0) > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            G._safe_torch_save({"epoch": epoch, "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), "val_metrics": val_metrics},
                               os.path.join(exp_dir, "best_model.pth"))

        if (epoch + 1) % 10 == 0:
            G._safe_torch_save({"epoch": epoch, "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict()},
                               os.path.join(exp_dir, f"checkpoint_epoch_{epoch+1:03d}.pth"))

        if config.EARLY_STOPPING_PATIENCE and (epoch + 1 - best_epoch) > config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Done: {(time.time()-t0)/60:.1f}min, Best F1={best_val_f1:.4f} @ ep {best_epoch}")
    if writer: writer.close()

if __name__ == "__main__":
    ckpt_path = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result\exp_hk\checkpoint_epoch_040.pth"
    resume_training(ckpt_path, target_epochs=100)
