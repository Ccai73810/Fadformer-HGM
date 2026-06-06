import os
import torch
import numpy as np
from PIL import Image

DATA_DIR = './Rain200H'
PRETRAINED = './pretrain_weights/rain200H/FADformer_Rain200H.pth'

print("="*70)
print("🔍 FADformer 基准测试 (Baseline) 异常排查工具")
print("="*70)

# 1. 检查权重文件
print("\n[1/4] 检查权重文件...")
if not os.path.exists(PRETRAINED):
    print(f"❌ 错误: 找不到权重文件 {PRETRAINED}")
    print("   请确保权重存放在该路径下，或者修改脚本中的 PRETRAINED 路径")
else:
    size_mb = os.path.getsize(PRETRAINED) / (1024*1024)
    print(f"✅ 找到权重文件: {PRETRAINED} (大小: {size_mb:.2f} MB)")
    try:
        ckpt = torch.load(PRETRAINED, map_location='cpu')
        print(f"   - 字典键: {list(ckpt.keys())}")
        state = ckpt.get('state_dict', ckpt)
        print(f"   - 参数项数量: {len(state)}")
        
        # 打印部分参数名称和形状
        print("   - 典型权重形状示例:")
        keys = list(state.keys())
        for k in keys[:3]:
            print(f"     • {k}: {list(state[k].shape)}")
    except Exception as e:
        print(f"❌ 错误: 读取权重失败: {e}")

# 2. 检查数据集结构和对齐
print("\n[2/4] 检查测试集文件对齐...")
test_input_dir = os.path.join(DATA_DIR, 'test', 'input')
test_target_dir = os.path.join(DATA_DIR, 'test', 'target')

if not os.path.exists(test_input_dir) or not os.path.exists(test_target_dir):
    print(f"❌ 错误: 测试集路径不存在!")
    print(f"   • 预期输入路径: {test_input_dir}")
    print(f"   • 预期目标路径: {test_target_dir}")
else:
    inputs = sorted([f for f in os.listdir(test_input_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
    targets = sorted([f for f in os.listdir(test_target_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
    
    print(f"✅ 找到输入 (Rainy) 图片: {len(inputs)} 张")
    print(f"✅ 找到目标 (Clean) 图片: {len(targets)} 张")
    
    if len(inputs) != len(targets):
        print(f"⚠️ 警告: 输入图片数量 ({len(inputs)}) 与目标图片数量 ({len(targets)}) 不一致!")
    
    if len(inputs) == 0:
        print("❌ 错误: 输入文件夹内没有图片!")
    else:
        print("\n   [文件名对齐检查] 前 5 个文件:")
        mismatch_count = 0
        for i in range(min(5, len(inputs), len(targets))):
            inp_name = inputs[i]
            tgt_name = targets[i]
            status = "✅ 相同" if inp_name == tgt_name else "❌ 不同"
            if inp_name != tgt_name:
                mismatch_count += 1
            print(f"     • 样本 {i:2d}: 输入 = {inp_name:<20} | 目标 = {tgt_name:<20} | {status}")
        
        if mismatch_count > 0:
            print("⚠️ 警告: 输入和目标文件名不一致！这会导致数据读取时错位（例如用 A 图的雨痕对比 B 图的无雨背景），从而导致 Baseline PSNR 极低。")

# 3. 检查图像内容和直连 PSNR
print("\n[3/4] 检查图像内容和直连 PSNR (不经过模型)...")
if os.path.exists(test_input_dir) and os.path.exists(test_target_dir) and len(inputs) > 0 and len(targets) > 0:
    try:
        # 读取第一张图
        inp_img = Image.open(os.path.join(test_input_dir, inputs[0])).convert('RGB')
        # 如果文件名不同，这里读取对应的第一个 target
        tgt_img = Image.open(os.path.join(test_target_dir, targets[0])).convert('RGB')
        
        inp_arr = np.array(inp_img).astype(np.float32) / 255.0
        tgt_arr = np.array(tgt_img).astype(np.float32) / 255.0
        
        if inp_arr.shape != tgt_arr.shape:
            print(f"⚠️ 警告: 样本 0 尺寸不一致! 输入形状: {inp_arr.shape}, 目标形状: {tgt_arr.shape}")
            # 进行简单裁剪以计算
            h = min(inp_arr.shape[0], tgt_arr.shape[0])
            w = min(inp_arr.shape[1], tgt_arr.shape[1])
            inp_arr = inp_arr[:h, :w, :]
            tgt_arr = tgt_arr[:h, :w, :]
            
        diff = inp_arr - tgt_arr
        mse = np.mean(diff ** 2)
        psnr_direct = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        mae = np.mean(np.abs(diff))
        
        print(f"   样本 0 分析:")
        print(f"   - 输入图片: {inputs[0]} (分辨率: {inp_img.size})")
        print(f"   - 目标图片: {targets[0]} (分辨率: {tgt_img.size})")
        print(f"   - 直连平均绝对误差 (MAE): {mae:.4f}")
        print(f"   - 直连 PSNR (无模型，纯雨图 vs 干净图): {psnr_direct:.2f} dB")
        
        if mse == 0:
            print("\n🚨 严重警告: 你的输入文件夹 (input) 和目标文件夹 (target) 里的图片完全一样！")
            print("   这意味着你的 `target` 文件夹里存放的其实也是有雨的图片。")
        elif psnr_direct > 35:
            print("\n⚠️ 警告: 直连 PSNR 极高，输入和目标非常相似，请检查是否放错了数据集。")
        else:
            print("\n   📊 直连 PSNR 在 18-22 dB 属于正常范围（雨痕引起的扰动）。")
            
    except Exception as e:
        print(f"❌ 读取或分析图像失败: {e}")

# 4. 模拟权重加载到基准模型
print("\n[4/4] 模拟权重加载并检查形状匹配...")
try:
    from models.FADformer import FADformer
    model = FADformer()
    model_state = model.state_dict()
    
    if os.path.exists(PRETRAINED):
        ckpt = torch.load(PRETRAINED, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        
        matched_keys = []
        mismatched_keys = []
        missing_keys = []
        
        for k, v in state.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            if name in model_state:
                if model_state[name].shape == v.shape:
                    matched_keys.append(name)
                else:
                    mismatched_keys.append((name, list(v.shape), list(model_state[name].shape)))
            else:
                missing_keys.append(name)
                
        print(f"   - 模型总参数项数: {len(model_state)}")
        print(f"   - 成功匹配的参数项数: {len(matched_keys)} / {len(model_state)}")
        print(f"   - 形状不匹配的参数项数: {len(mismatched_keys)}")
        print(f"   - 权重中有多余的参数项数: {len(missing_keys)}")
        
        if len(mismatched_keys) > 0:
            print(f"🚨 警告: 发现 {len(mismatched_keys)} 个参数形状不匹配！前 3 个为:")
            for m in mismatched_keys[:3]:
                print(f"     • {m[0]}: 权重形状={m[1]} | 模型形状={m[2]}")
            print("   这通常是因为你将 Full 模型的权重加载到了 Mini 模型中，或者反之。")
        elif len(matched_keys) == len(model_state):
            print("✅ 完美加载！所有 Backbone 权重与模型完美契合，未发生任何形状冲突。")
        else:
            # 有一些是 HGM 特有的，但 Backbone 的应该全部加载
            backbone_missing = [k for k in model_state.keys() if k not in matched_keys and 'hgm' not in k.lower() and 'ca_conv' not in k.lower()]
            if backbone_missing:
                print(f"⚠️ 警告: 以下 Backbone 参数未被成功加载 (共 {len(backbone_missing)} 项):")
                print(f"     • {backbone_missing[:5]}...")
            else:
                print("✅ 加载正常：原始 Backbone 参数已全部成功加载（没有被过滤）。")

except Exception as e:
    print(f"❌ 模拟权重加载失败: {e}")

print("\n" + "="*70)
print("📌 排查建议:")
print("1. 如果直连 PSNR 也是 18.58 dB 左右，说明模型根本没有起到去雨作用（可能是模型没加载上权重或输入被原样输出）。")
print("2. 检查 `test/input` 和 `test/target` 的文件数量和文件名是否 100% 一一对应并对齐。")
print("3. 如果文件名对齐但 PSNR 仍低，请人工查看第一组图像（可以用 python 另存一张），确认其是否是配对的（即同一张图的有雨和无雨版本）。")
print("="*70)
