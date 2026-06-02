# fusion/train_tcn.py — P1.2: Train 2A TCN prior predictor
# ==========================================================
# Uses Module 1 MoG outputs as soft labels for NLOS prediction.
# Trains one TCN per dataset.
# ==========================================================
import sys, os, pickle, numpy as np, torch, torch.nn as nn, time

_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _MODEL_DIR)
sys.path.insert(0, r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model")

from fusion.motion_geometry_predictor import MotionGeometryPredictor
from fusion.utils import load_epoch_data, load_mog_model, run_mog_inference

# ---- Config ----
MAX_SV = 20
SEQ_LEN = 10
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODELS_DIR = os.path.join(os.path.dirname(_MODEL_DIR), 'models')
CACHE_DIR = os.path.join(os.path.dirname(_MODEL_DIR), 'cache')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DATASETS = {
    'berlin1_potsdamer_platz': 'exp_034',
    'berlin2_gendarmenmarkt': 'exp_035',
    'frankfurt1_maintower': 'exp_036',
    'frankfurt2_westendtower': 'exp_037',
}


def build_sequences(dataset_name, exp_name, max_epochs=None):
    '''Build training sequences from Module 1 outputs.'''
    cache_path = os.path.join(CACHE_DIR, f'{dataset_name}_tcn_data.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f: return pickle.load(f)
    
    print(f'  Building sequences for {dataset_name}...')
    all_epochs = load_epoch_data(dataset_name)
    if max_epochs: all_epochs = all_epochs[:max_epochs]
    
    # Run Module 1 inference on all epochs
    model, config, device = load_mog_model(exp_name)
    mog_outputs = []
    for i, ep in enumerate(all_epochs):
        mog = run_mog_inference(model, config, device, ep)
        mog_outputs.append(mog)
        if (i + 1) % 500 == 0: print(f'    Inference: {i+1}/{len(all_epochs)}')
    
    # Build sequences
    X, Y, masks = [], [], []
    for t in range(SEQ_LEN, len(all_epochs)):
        seq_feats = []
        for offset in range(SEQ_LEN, 0, -1):
            ep_idx = t - offset
            ep = all_epochs[ep_idx]
            mog = mog_outputs[ep_idx]
            if mog is None: continue
            
            # Features per timestep: velocity + geometry
            if ep_idx > 0:
                vel = all_epochs[ep_idx]['gt_ecef'] - all_epochs[ep_idx-1]['gt_ecef']
            else:
                vel = np.zeros(3)
            
            geom = np.zeros((MAX_SV, 3))  # elevation/90, azimuth/360, p_los
            for j in range(min(len(mog['elevation_deg']), MAX_SV)):
                geom[j, 0] = mog['elevation_deg'][j] / 90.0
                geom[j, 1] = mog['azimuth_deg'][j] / 360.0
                geom[j, 2] = mog.get('p_los_sharp', mog['p_los'])[j]
            
            ts = np.concatenate([vel, geom.flatten()])  # 3 + 20*3 = 63
            seq_feats.append(ts)
        
        if len(seq_feats) < SEQ_LEN: continue
        X.append(np.stack(seq_feats, axis=0))  # (SEQ_LEN, 63)
        
        # Target: p_nlos for epoch t
        target_mog = mog_outputs[t]
        if target_mog is None: continue
        p_los_t = target_mog.get('p_los_sharp', target_mog['p_los'])
        target = 1.0 - p_los_t  # p_nlos
        vis_mask = np.zeros(MAX_SV)
        N_vis = min(len(target), MAX_SV)
        vis_mask[:N_vis] = 1.0
        target_padded = np.zeros(MAX_SV)
        target_padded[:N_vis] = target[:N_vis]
        Y.append(target_padded)
        masks.append(vis_mask)
    
    data = {
        'X': np.array(X, dtype=np.float32),      # (N_seq, SEQ_LEN, 63)
        'Y': np.array(Y, dtype=np.float32),       # (N_seq, MAX_SV)
        'masks': np.array(masks, dtype=np.float32),  # (N_seq, MAX_SV)
    }
    with open(cache_path, 'wb') as f: pickle.dump(data, f)
    print(f'    Saved {len(X)} sequences to cache')
    return data


class SimpleTCN(nn.Module):
    '''Lightweight TCN for p_nlos prediction.'''
    def __init__(self, in_dim=63, hidden=64, out_dim=20, seq_len=10):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        # 1D convs over time (kernel=3, dilation=1,2,4)
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ln3 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (B, seq_len, in_dim)
        B, T, D = x.shape
        h = self.input_proj(x)  # (B, T, hidden)
        h = h.permute(0, 2, 1)  # (B, hidden, T) for Conv1d
        h = self.ln1(h.permute(0, 2, 1)).permute(0, 2, 1)  # LayerNorm over hidden
        h = h + torch.relu(self.conv1(h))  # residual block 1
        h = self.ln2(h.permute(0, 2, 1)).permute(0, 2, 1)  # LayerNorm over hidden
        h = h + torch.relu(self.conv2(h))  # residual block 2
        h = h + torch.relu(self.conv3(h))  # residual block 3
        h = h.permute(0, 2, 1)  # (B, T, hidden)
        h = self.ln3(self.dropout(h))
        h_last = h[:, -1, :]    # last timestep
        return torch.sigmoid(self.out(h_last))  # (B, MAX_SV)


def train_tcn(dataset_name, exp_name, max_epochs=None):
    print(f'\n===== Training TCN for {dataset_name} =====')
    data = build_sequences(dataset_name, exp_name, max_epochs)
    
    X = torch.tensor(data['X']); Y = torch.tensor(data['Y']); M = torch.tensor(data['masks'])
    n = len(X); n_train = int(n * 0.8)
    idx = torch.randperm(n)
    X_train, Y_train, M_train = X[idx[:n_train]], Y[idx[:n_train]], M[idx[:n_train]]
    X_val, Y_val, M_val = X[idx[n_train:]], Y[idx[n_train:]], M[idx[n_train:]]
    
    model = SimpleTCN(63, HIDDEN_DIM, MAX_SV, SEQ_LEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss(reduction='none')
    
    best_val = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            xb = X_train[i:i+BATCH_SIZE].to(DEVICE)
            yb = Y_train[i:i+BATCH_SIZE].to(DEVICE)
            mb = M_train[i:i+BATCH_SIZE].to(DEVICE)
            pred = model(xb)
            loss = (loss_fn(pred, yb) * mb).sum() / mb.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            xv = X_val.to(DEVICE); yv = Y_val.to(DEVICE); mv = M_val.to(DEVICE)
            pred_v = model(xv)
            val_loss = (loss_fn(pred_v, yv) * mv).sum() / mv.sum()
        
        if val_loss < best_val:
            best_val = val_loss
            save_path = os.path.join(MODELS_DIR, f'tcn_{dataset_name}.pth')
            torch.save(model.state_dict(), save_path)
        
        if epoch % 5 == 0:
            print(f'  Epoch {epoch}: train_loss={total_loss/((n_train+BATCH_SIZE-1)//BATCH_SIZE):.4f}, val_loss={val_loss.item():.4f}')
    
    print(f'  Best val_loss={best_val:.4f}, saved to {save_path}')
    return save_path


if __name__ == '__main__':
    print('TCN Trainer for Module 2A')
    print(f'Device: {DEVICE}')
    for ds, exp in DATASETS.items():
        train_tcn(ds, exp, max_epochs=500)  # limit for speed
    print('\nAll TCN models trained!')
