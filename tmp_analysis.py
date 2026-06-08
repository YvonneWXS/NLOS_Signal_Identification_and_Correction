import json, os
base = r'D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result'
datasets = ['berlin1_potsdamer_platz','berlin2_gendarmenmarkt','frankfurt1_maintower','frankfurt2_westendtower']
for ds in datasets:
    path = os.path.join(base, f'analysis_{ds}.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f'\n=== {ds} ===')
        for k in ['experiment','accuracy','f1','precision','recall','plos_gap','p_los_los_mean','p_los_nlos_mean']:
            if k in data:
                v = data[k]
                if isinstance(v,float): print(f'  {k}: {v:.4f}')
                else: print(f'  {k}: {v}')
        if 'classification' in data:
            c = data['classification']
            print(f'  TP={c.get("TP","?")} FP={c.get("FP","?")} TN={c.get("TN","?")} FN={c.get("FN","?")}')
        if 'sigma_nlos_mean' in data:
            print(f'  sigma_los_mean: {data.get("sigma_los_mean","?"):.4f}')
            print(f'  sigma_nlos_mean: {data.get("sigma_nlos_mean","?"):.4f}')
        if 'mu_nlos_los' in data:
            print(f'  mu_LOS: {data["mu_nlos_los"]:.0f}m  mu_NLOS: {data["mu_nlos_nlos"]:.0f}m')
    else:
        print(f'{ds}: NO ANALYSIS FILE')
print('\nDone')
