import os
import sys
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from models.FADformer import FADformer_mini, FADformer_HGM_mini

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on Device: {device}")
    
    model_mini = FADformer_mini().to(device)
    model_hgm = FADformer_HGM_mini(window_size=8, num_heads=4, fusion_mode='gate').to(device)
    
    model_mini.eval()
    model_hgm.eval()
    
    x = torch.randn(1, 3, 256, 256, device=device)
    
    params_mini = sum(p.numel() for p in model_mini.parameters())
    params_hgm = sum(p.numel() for p in model_hgm.parameters())
    
    print(f"FADformer-mini Parameters:      {params_mini/1e6:.4f} M")
    print(f"FADformer-HGM-mini Parameters:  {params_hgm/1e6:.4f} M")
    
    if HAS_THOP:
        macs_mini, _ = profile(model_mini, inputs=(x,), verbose=False)
        macs_hgm, _ = profile(model_hgm, inputs=(x,), verbose=False)
        # MACs * 2 ≈ FLOPs (Multiply-Accumulate is two operations)
        print(f"FADformer-mini MACs (256x256):       {macs_mini/1e9:.4f} G")
        print(f"FADformer-HGM-mini MACs (256x256):   {macs_hgm/1e9:.4f} G")
    else:
        print("thop package not installed. Cannot measure MACs/FLOPs.")

if __name__ == '__main__':
    main()
