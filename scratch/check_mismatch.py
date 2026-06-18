import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from models.FADformer import FADformer_HGM, FADformer
import torch

def check(model, label):
    ckpt = torch.load('./pretrain_weights/rain200H/FADformer_Rain200H.pth', map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    model_state = model.state_dict()
    mismatches = []
    matched = 0
    missing = []
    
    for k, v in state.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        if name in model_state:
            if model_state[name].shape == v.shape:
                matched += 1
            else:
                mismatches.append((name, list(v.shape), list(model_state[name].shape)))
        else:
            missing.append(name)
            
    print(f"=== {label} ===")
    print(f"Total keys in ckpt: {len(state)}")
    print(f"Total keys in model: {len(model_state)}")
    print(f"Matched: {matched}")
    print(f"Mismatches: {len(mismatches)}")
    if mismatches:
        print("First 5 mismatches:", mismatches[:5])
    print(f"Missing in model: {len(missing)}")

check(FADformer(), "Original FADformer (Full)")
check(FADformer_HGM(), "FADformer_HGM (Full)")
