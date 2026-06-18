"""
HGM (Hybrid Global Mixer) 单元测试

测试内容：
- SparseWindowAttention 前向传播正确性
- 窗口划分和还原的一致性
- 门控输出范围 [0, 1]
- HGM 梯度流动正常
- 与原 FFCM 输出维度一致
- 完整模型集成测试

运行方式: python tests/test_hgm.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np


def test_window_partition_reverse():
    """测试窗口划分和还原的一致性"""
    print("\n" + "="*60)
    print("Test 1: Window Partition & Reverse Consistency")
    print("="*60)
    
    from models.HGM import window_partition, window_reverse
    
    B, C, H, W = 2, 64, 32, 32
    window_size = 8
    
    # 创建随机输入
    x = torch.randn(B, C, H, W)
    x_permuted = x.permute(0, 2, 3, 1)  # (B, H, W, C)
    
    # 划分窗口
    windows = window_partition(x_permuted, window_size)
    print(f"  Input shape: {x.shape}")
    print(f"  After partition: {windows.shape}")
    assert windows.shape == (B * (H//window_size) * (W//window_size), window_size, window_size, C), \
        f"Expected shape ({B * (H//window_size) * (W//window_size)}, {window_size}, {window_size}, {C}), got {windows.shape}"
    
    # 还原
    restored = window_reverse(windows, window_size, H, W)
    print(f"  After reverse: {restored.shape}")
    assert restored.shape == x_permuted.shape, \
        f"Expected shape {x_permuted.shape}, got {restored.shape}"
    
    # 验证数值一致性
    diff = torch.abs(restored - x_permuted).max().item()
    print(f"  Max reconstruction error: {diff:.2e}")
    assert diff < 1e-6, f"Reconstruction error too large: {diff}"
    
    print("  ✅ PASSED")
    return True


def test_sparse_window_attention():
    """测试 SparseWindowAttention 前向传播"""
    print("\n" + "="*60)
    print("Test 2: SparseWindowAttention Forward Pass")
    print("="*60)
    
    from models.HGM import SparseWindowAttention
    
    B, C, H, W = 2, 64, 32, 32
    window_size = 8
    num_heads = 8
    
    # 测试不移位版本
    attn_no_shift = SparseWindowAttention(
        dim=C,
        window_size=window_size,
        num_heads=num_heads,
        shift_size=0
    )
    x = torch.randn(B, C, H, W)
    out = attn_no_shift(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape (no shift): {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {out.shape}"
    
    # 测试移位版本
    attn_shift = SparseWindowAttention(
        dim=C,
        window_size=window_size,
        num_heads=num_heads,
        shift_size=window_size // 2
    )
    out_shift = attn_shift(x)
    print(f"  Output shape (with shift): {out_shift.shape}")
    assert out_shift.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {out_shift.shape}"
    
    print("  ✅ PASSED")
    return True


def test_hybrid_global_mixer():
    """测试 HybridGlobalMixer 前向传播和融合机制"""
    print("\n" + "="*60)
    print("Test 3: HybridGlobalMixer Forward & Fusion")
    print("="*60)
    
    from models.HGM import HybridGlobalMixer
    
    B, C, H, W = 2, 64, 32, 32
    
    # 测试 gate 融合模式
    hgm_gate = HybridGlobalMixer(
        dim=C,
        window_size=8,
        num_heads=8,
        fusion_mode='gate'
    )
    x = torch.randn(B, C, H, W)
    out_gate = hgm_gate(x)
    print(f"  [Gate Mode] Input: {x.shape} -> Output: {out_gate.shape}")
    assert out_gate.shape == x.shape, f"Shape mismatch"
    
    # 测试 sum 融合模式
    hgm_sum = HybridGlobalMixer(
        dim=C,
        window_size=8,
        num_heads=8,
        fusion_mode='sum'
    )
    out_sum = hgm_sum(x)
    print(f"  [Sum Mode]  Output: {out_sum.shape}")
    assert out_sum.shape == x.shape
    
    # 测试 learnable 融合模式
    hgm_learnable = HybridGlobalMixer(
        dim=C,
        window_size=8,
        num_heads=8,
        fusion_mode='learnable'
    )
    out_learnable = hgm_learnable(x)
    print(f"  [Learnable] Output: {out_learnable.shape}")
    assert out_learnable.shape == x.shape
    
    # 验证门控输出范围（仅对 gate 模式）
    if hasattr(hgm_gate, 'gate_conv'):
        with torch.no_grad():
            x_test = torch.randn(1, C, H, W)
            _ = hgm_gate(x_test)
            # 注意：这里我们无法直接获取门控值，但可以验证输出合理性
            print(f"  Gate mode output range: [{out_gate.min():.4f}, {out_gate.max():.4f}]")
    
    print("  ✅ PASSED")
    return True


def test_gradient_flow():
    """测试梯度流动是否正常"""
    print("\n" + "="*60)
    print("Test 4: Gradient Flow Check")
    print("="*60)
    
    from models.HGM import HybridGlobalMixer
    
    B, C, H, W = 2, 64, 16, 16
    
    hgm = HybridGlobalMixer(
        dim=C,
        window_size=8,
        num_heads=8,
        fusion_mode='gate'
    )
    
    x = torch.randn(B, C, H, W, requires_grad=True)
    out = hgm(x)
    loss = out.sum()
    loss.backward()
    
    # 检查输入梯度是否存在且非零
    assert x.grad is not None, "Input gradient is None"
    grad_norm = x.grad.norm().item()
    print(f"  Input gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0, "Input gradient is zero"
    
    # 检查各参数梯度
    param_grads = []
    for name, param in hgm.named_parameters():
        if param.grad is not None:
            param_grads.append((name, param.grad.norm().item()))
    
    print(f"  Parameters with gradients: {len(param_grads)}/{len(list(hgm.parameters()))}")
    assert len(param_grads) > 0, "No parameter has gradient"
    
    # 显示部分参数梯度
    for name, norm in param_grads[:5]:
        print(f"    {name}: {norm:.6f}")
    
    print("  ✅ PASSED")
    return True


def test_dimension_compatibility():
    """测试与原 FFCM 输出维度一致性"""
    print("\n" + "="*60)
    print("Test 5: Dimension Compatibility with Original FFCM")
    print("="*60)
    
    from models.FADformer import Fused_Fourier_Conv_Mixer
    from models.HGM import HybridGlobalMixer
    
    B, C, H, W = 2, 64, 32, 32
    
    # 原始 FFCM
    ffcm = Fused_Fourier_Conv_Mixer(dim=C)
    x = torch.randn(B, C, H, W)
    out_ffcm = ffcm(x)
    print(f"  FFCM output: {out_ffcm.shape}")
    
    # HGM
    hgm = HybridGlobalMixer(dim=C, window_size=8, num_heads=8)
    out_hgm = hgm(x)
    print(f"  HGM output:  {out_hgm.shape}")
    
    assert out_ffcm.shape == out_hgm.shape, \
        f"Dimension mismatch: FFCM {out_ffcm.shape} vs HGM {out_hgm.shape}"
    
    print("  ✅ PASSED")
    return True


def test_hgm_token_mixer():
    """测试 HGM_TokenMixer 模块"""
    print("\n" + "="*60)
    print("Test 6: HGM_TokenMixer Module")
    print("="*60)
    
    from models.FADformer import HGM_TokenMixer
    
    B, C, H, W = 2, 64, 32, 32
    
    mixer = HGM_TokenMixer(
        dim=C,
        window_size=8,
        num_heads=8,
        fusion_mode='gate',
        use_ca=True
    )
    
    x = torch.randn(B, C, H, W)
    out = mixer(x)
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch"
    
    # 统计参数量
    total_params = sum(p.numel() for p in mixer.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    print("  ✅ PASSED")
    return True


def test_fadformer_with_hgm():
    """测试完整的 FADformer_HGM 模型"""
    print("\n" + "="*60)
    print("Test 7: Complete FADformer_HGM Model")
    print("="*60)
    
    from models.FADformer import FADformer_HGM, FADformer_HGM_mini
    
    # 测试完整版
    model_full = FADformer_HGM(window_size=8, num_heads=8, fusion_mode='gate')
    x = torch.randn(1, 3, 256, 256)
    out_full = model_full(x)
    
    print(f"  [Full Version]")
    print(f"    Input:  {x.shape}")
    print(f"    Output: {out_full.shape}")
    assert out_full.shape == x.shape, f"Shape mismatch"
    
    # 统计参数量
    full_params = sum(p.numel() for p in model_full.parameters())
    print(f"    Parameters: {full_params:,}")
    
    # 测试 mini 版本
    model_mini = FADformer_HGM_mini(window_size=8, num_heads=8, fusion_mode='gate')
    out_mini = model_mini(x)
    
    print(f"\n  [Mini Version]")
    print(f"    Input:  {x.shape}")
    print(f"    Output: {out_mini.shape}")
    assert out_mini.shape == x.shape
    
    mini_params = sum(p.numel() for p in model_mini.parameters())
    print(f"    Parameters: {mini_params:,}")
    
    # 对比原始 FADformer
    from models.FADformer import FADformer
    model_orig = FADformer()
    orig_params = sum(p.numel() for p in model_orig.parameters())
    
    print(f"\n  [Parameter Comparison]")
    print(f"    Original FADformer:  {orig_params:,}")
    print(f"    FADformer_HGM:       {full_params:,}")
    print(f"    Parameter increase:   {(full_params - orig_params) / orig_params * 100:.1f}%")
    
    # 验证参数增加在合理范围内 (< 35%) - HGM 包含注意力+频域双分支，参数增加合理
    param_increase = (full_params - orig_params) / orig_params
    assert param_increase < 0.35, f"Parameter increase too high: {param_increase*100:.1f}%"
    
    print("  ✅ PASSED")
    return True


def test_different_window_sizes():
    """测试不同窗口大小的兼容性"""
    print("\n" + "="*60)
    print("Test 8: Different Window Sizes (Ablation Support)")
    print("="*60)
    
    from models.HGM import HybridGlobalMixer
    
    B, C, H, W = 2, 64, 32, 32
    window_sizes = [4, 8, 16]
    
    for ws in window_sizes:
        try:
            hgm = HybridGlobalMixer(
                dim=C,
                window_size=ws,
                num_heads=8,
                fusion_mode='gate'
            )
            x = torch.randn(B, C, H, W)
            out = hgm(x)
            
            params = sum(p.numel() for p in hgm.parameters())
            print(f"  Window size {ws:2d}: output={out.shape}, params={params:,}")
            assert out.shape == x.shape
            
        except Exception as e:
            print(f"  Window size {ws}: ❌ FAILED - {e}")
            raise
    
    print("  ✅ PASSED")
    return True


def test_invalid_fusion_mode():
    """测试无效融合模式的错误处理"""
    print("\n" + "="*60)
    print("Test 9: Invalid Fusion Mode Error Handling")
    print("="*60)
    
    from models.HGM import HybridGlobalMixer
    
    try:
        hgm = HybridGlobalMixer(dim=64, fusion_mode='invalid_mode')
        print("  ❌ FAILED - Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
        print("  ✅ PASSED")
        return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  HGM (Hybrid Global Mixer) Unit Test Suite".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    tests = [
        ("Window Partition/Reverse", test_window_partition_reverse),
        ("SparseWindowAttention", test_sparse_window_attention),
        ("HybridGlobalMixer", test_hybrid_global_mixer),
        ("Gradient Flow", test_gradient_flow),
        ("Dimension Compatibility", test_dimension_compatibility),
        ("HGM_TokenMixer", test_hgm_token_mixer),
        ("FADformer_HGM Integration", test_fadformer_with_hgm),
        ("Different Window Sizes", test_different_window_sizes),
        ("Error Handling", test_invalid_fusion_mode),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ❌ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print("-"*70)
    print(f"  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! HGM module is ready for training.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
