import torch, os
base = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result"
for e in ["exp_030","exp_031","exp_032","exp_033"]:
    ckpt = os.path.join(base, e, "best_model.pth")
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = d["val_metrics"]
    print("=== {} (epoch {}) ===".format(e, d["epoch"]))
    for k in ["f1","accuracy","precision","recall","p_los_los_avg","p_los_nlos_avg","mu_nlos_mean","sigma_los_mean","sigma_sep"]:
        print("  {}: {:.4f}".format(k, m.get(k, 0)))
