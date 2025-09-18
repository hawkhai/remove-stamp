#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
印章擦除系统 - 统一示例代码
基于GAN的端到端印章/水印/噪声去除系统

功能:
- 快速演示模式
- 单张图像处理
- 批量图像处理
- 依赖检查

使用方法:
1. 快速演示: python example.py
2. 单张处理: python example.py --input_image image.jpg
3. 批量处理: python example.py --input_dir images/ --output_dir results/
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from models.sa_gan import STRnet2
import cv2
from skimage import color


def check_dependencies():
    """检查必要的依赖是否安装"""
    print("🔍 检查依赖...")
    deps_ok = True
    
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch 未安装")
        deps_ok = False
    
    try:
        import torchvision
        print(f"   ✅ TorchVision: {torchvision.__version__}")
    except ImportError:
        print("   ❌ TorchVision 未安装")
        deps_ok = False
    
    try:
        from PIL import Image
        print("   ✅ PIL/Pillow")
    except ImportError:
        print("   ❌ PIL/Pillow 未安装")
        deps_ok = False
    
    try:
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
    except ImportError:
        print("   ❌ NumPy 未安装")
        deps_ok = False
    
    try:
        import cv2
        print(f"   ✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("   ❌ OpenCV 未安装")
        deps_ok = False
    
    try:
        import skimage
        print(f"   ✅ scikit-image: {skimage.__version__}")
    except ImportError:
        print("   ❌ scikit-image 未安装")
        deps_ok = False
    
    if not deps_ok:
        print("\n❌ 请先安装必要的依赖:")
        print("pip install torch torchvision pillow numpy opencv-python scikit-image")
    
    return deps_ok


def load_image(image_path, load_size=(512, 512)):
    """加载并预处理图像 - 修复：移除错误的归一化"""
    transform = transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),  # 只转换到[0,1]，不做[-1,1]归一化
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def save_result(tensor, save_path):
    """保存处理结果 - 修复：移除错误的反归一化"""
    # 模型输出已经在[0,1]范围内，不需要反归一化
    tensor = torch.clamp(tensor, 0, 1)
    save_image(tensor, save_path)


def create_demo_image():
    """创建演示图像"""
    img = Image.new('RGB', (512, 512), 'white')
    # 可以在这里添加更复杂的演示图像生成逻辑
    return img


def process_model_mask(model_mask, target_size, enhance_quality=True, keep_grayscale=True):
    """
    处理模型生成的mask张量 - 支持灰度和二值化
    
    Args:
        model_mask: torch.Tensor - 模型生成的mask
        target_size: tuple - 目标尺寸 (width, height)
        enhance_quality: bool - 是否进行质量改进后处理
        keep_grayscale: bool - 是否保留灰度信息（避免锯齿）
    
    Returns:
        numpy.ndarray - 处理后的mask (0-255)
    """
    # 处理张量维度
    if model_mask.dim() == 4:
        model_mask = model_mask.squeeze(0)
    if model_mask.dim() == 3:
        model_mask = model_mask.mean(dim=0)  # 多通道取平均
    
    # 转换为numpy数组
    mask_np = model_mask.detach().cpu().numpy()
    
    # 调整尺寸
    if mask_np.shape != target_size[::-1]:  # target_size是(w,h)，需要转换为(h,w)
        mask_np = cv2.resize(mask_np, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 归一化到[0,255]
    mask_min, mask_max = mask_np.min(), mask_np.max()
    if mask_max > mask_min:
        mask_np = (mask_np - mask_min) / (mask_max - mask_min) * 255
    else:
        mask_np = np.zeros_like(mask_np) * 255
    
    mask_np = mask_np.astype(np.uint8)
    
    if keep_grayscale:
        # 保留灰度信息，避免锯齿
        # 检查原始mask的平均值来判断需要如何处理方向
        mask_mean = np.mean(mask_np)
        
        # 重新判断mask方向：通常模型输出中，印章区域应该是高值
        # 如果原始mask平均值较大，说明大部分是高值（可能是背景）
        if mask_mean > 127.5:  # 原始mask偏亮，可能需要反转
            processed_mask = 255 - mask_np  # 反转，让印章区域变为低值
        else:
            processed_mask = mask_np
        
        # 灰度模式的质量改进
        if enhance_quality:
            # 1. 高斯模糊平滑边缘，避免锯齿
            processed_mask = cv2.GaussianBlur(processed_mask, (3, 3), 0.5)
            
            # 2. 轻微的对比度增强
            processed_mask = cv2.convertScaleAbs(processed_mask, alpha=1.1, beta=0)
            
            # 3. 确保在[0,255]范围内
            processed_mask = np.clip(processed_mask, 0, 255)
    
    else:
        # 传统二值化模式
        _, binary_mask = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 检查原始mask的平均值来判断需要如何处理
        mask_mean = np.mean(mask_np)
        
        # 如果原始mask平均值较小，说明印章区域是低值
        if mask_mean < 127.5:  # 原始mask偏暗，印章区域是低值
            binary_mask = 255 - binary_mask  # 反转，让印章区域变为白色
        
        # 二值化模式的质量改进
        if enhance_quality:
            # 1. 去除小噪声点
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
            
            # 2. 填充小空洞
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_small)
            
            # 3. 平滑边缘
            binary_mask = cv2.medianBlur(binary_mask, 3)
            
            # 4. 轻微膨胀，确保印章区域完整
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            binary_mask = cv2.dilate(binary_mask, kernel_dilate, iterations=1)
        
        processed_mask = binary_mask
    
    return processed_mask.astype(np.uint8)


def extract_stamp_and_mask(original_image, cleaned_image, model_mask, enhance_mask=True, use_grayscale=True):
    """
    基于模型mask的印章提取算法 - 支持灰度和二值化
    
    Args:
        original_image: PIL Image - 原始带印章图像
        cleaned_image: PIL Image - 擦除印章后的图像
        model_mask: torch.Tensor - 模型生成的印章区域mask
        enhance_mask: bool - 是否进行mask质量改进
        use_grayscale: bool - 是否使用灰度mask（避免锯齿）
    
    Returns:
        stamp_image: PIL Image - 提取的印章图像
        mask_image: PIL Image - 印章区域mask
        stats: dict - 提取统计信息
    """
    # 确保图像尺寸一致
    if original_image.size != cleaned_image.size:
        cleaned_image = cleaned_image.resize(original_image.size, Image.BICUBIC)
    
    # 转换为numpy数组
    orig_np = np.array(original_image)
    clean_np = np.array(cleaned_image)
    
    # 处理模型mask
    processed_mask = process_model_mask(model_mask, original_image.size, 
                                       enhance_quality=enhance_mask, 
                                       keep_grayscale=use_grayscale)
    
    # 提取印章
    stamp_np = orig_np.copy()
    
    if use_grayscale:
        # 灰度模式：使用软混合，避免锯齿
        mask_weight = processed_mask.astype(np.float32) / 255.0
        
        # 现在processed_mask中：低值=印章区域，高值=背景区域
        # 使用阈值来决定哪些区域设为白色背景
        background_threshold = 0.5  # 使用中值作为阈值
        background_mask = mask_weight > background_threshold  # 高值区域为背景
        
        # 将背景区域设为白色，保留印章区域
        stamp_np[background_mask] = [255, 255, 255]
    else:
        # 二值化模式：硬边缘
        # 需要重新检查二值化后的mask方向
        stamp_np[processed_mask > 127] = [255, 255, 255]  # 将高值区域设为白色背景
    
    # 转换为PIL图像
    stamp_image = Image.fromarray(stamp_np)
    mask_image = Image.fromarray(processed_mask, mode='L')
    
    # 计算统计信息
    total_pixels = processed_mask.shape[0] * processed_mask.shape[1]
    
    if use_grayscale:
        # 灰度模式：统计低值区域作为印章
        stamp_pixels = np.sum(processed_mask <= 127)  # 低于等于中值的像素为印章
    else:
        # 二值化模式：统计低值区域作为印章
        stamp_pixels = np.sum(processed_mask <= 127)
    
    stamp_ratio = stamp_pixels / total_pixels if total_pixels > 0 else 0
    
    # 计算原图和清理图的LAB差异（用于质量评估）
    try:
        orig_lab = color.rgb2lab(orig_np / 255.0)
        clean_lab = color.rgb2lab(clean_np / 255.0)
        lab_diff = np.sqrt(np.sum((orig_lab - clean_lab) ** 2, axis=2))
        max_lab_diff = np.max(lab_diff)
        mean_lab_diff = np.mean(lab_diff)
        
        # 计算印章区域的平均LAB差异
        stamp_lab_diff = np.mean(lab_diff[processed_mask <= 127]) if stamp_pixels > 0 else 0
    except Exception:
        max_lab_diff = mean_lab_diff = stamp_lab_diff = 0
    
    stats = {
        'total_pixels': total_pixels,
        'stamp_pixels': stamp_pixels,
        'stamp_ratio': stamp_ratio,
        'max_lab_diff': max_lab_diff,
        'mean_lab_diff': mean_lab_diff,
        'stamp_lab_diff': stamp_lab_diff,
        'method': 'model_mask_grayscale' if use_grayscale else 'model_mask_binary'
    }
    
    return stamp_image, mask_image, stats


def ensure_dir(file_path):
    """确保文件路径的目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def tensor_to_pil(tensor):
    """将tensor转换为PIL图像"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 确保在[0,1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为numpy并调整维度顺序
    np_array = tensor.permute(1, 2, 0).numpy()
    np_array = (np_array * 255).astype(np.uint8)
    
    return Image.fromarray(np_array)


def compute_image_quality_metrics(image_tensor, original_tensor):
    """
    计算图像质量指标
    
    Args:
        image_tensor: 处理后的图像张量
        original_tensor: 原始图像张量
    
    Returns:
        dict: 质量指标字典
    """
    # 转换为numpy进行计算
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    if original_tensor.dim() == 4:
        original_tensor = original_tensor.squeeze(0)
    
    # 确保两个张量尺寸一致
    if image_tensor.shape != original_tensor.shape:
        # 将image_tensor调整到original_tensor的尺寸
        target_size = original_tensor.shape[1:]  # (H, W)
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
        ).squeeze(0)
    
    img_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    orig_np = original_tensor.permute(1, 2, 0).detach().cpu().numpy()
    
    # 确保在[0,1]范围内
    img_np = np.clip(img_np, 0, 1)
    orig_np = np.clip(orig_np, 0, 1)
    
    # 1. MSE (越小越好)
    mse = np.mean((img_np - orig_np) ** 2)
    
    # 2. PSNR (越大越好)
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # 3. 结构相似性 (简化版SSIM)
    def simple_ssim(x, y):
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.var(x)
        sigma_y = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        return ssim
    
    ssim = simple_ssim(img_np, orig_np)
    
    # 4. 边缘保持度 (计算梯度相似性)
    def compute_gradient_magnitude(img):
        gray = np.mean(img, axis=2) if len(img.shape) == 3 else img
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        return np.sqrt(grad_x**2 + grad_y**2)
    
    grad_orig = compute_gradient_magnitude(orig_np)
    grad_img = compute_gradient_magnitude(img_np)
    edge_preservation = np.corrcoef(grad_orig.flatten(), grad_img.flatten())[0, 1]
    if np.isnan(edge_preservation):
        edge_preservation = 0
    
    # 5. 颜色一致性 (LAB空间差异)
    try:
        orig_lab = color.rgb2lab(orig_np)
        img_lab = color.rgb2lab(img_np)
        color_diff = np.mean(np.sqrt(np.sum((orig_lab - img_lab) ** 2, axis=2)))
    except:
        color_diff = 0
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'edge_preservation': edge_preservation,
        'color_diff': color_diff,
        'quality_score': psnr * 0.4 + ssim * 30 + edge_preservation * 20 - color_diff * 2
    }


def select_best_output_combination(outputs, original_tensor, masks, verbose=False):
    """
    从多个模型输出中选择最佳组合
    
    Args:
        outputs: dict - 包含所有模型输出的字典
        original_tensor: 原始图像张量
        masks: dict - 包含不同mask的字典
        verbose: 是否打印详细信息
    
    Returns:
        tuple: (最佳清理图像, 最佳mask, 质量统计)
    """
    # 获取目标尺寸（使用原始输入的尺寸）
    if original_tensor.dim() == 4:
        target_size = original_tensor.shape[2:]  # (H, W)
    else:
        target_size = original_tensor.shape[1:]  # (H, W)
    
    # 将所有输出调整到相同尺寸
    candidates = {}
    for name, output in outputs.items():
        if output.dim() == 4:
            output = output.squeeze(0)
        
        # 调整尺寸
        if output.shape[1:] != target_size:
            output = torch.nn.functional.interpolate(
                output.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
            ).squeeze(0)
        
        candidates[{
            'out1': 'scale1_output',
            'out2': 'scale2_output', 
            'out3': 'unet_output',
            'g_images': 'final_refined'
        }.get(name, name)] = output
    
    # 计算每个候选输出的质量指标
    quality_results = {}
    for name, output in candidates.items():
        try:
            metrics = compute_image_quality_metrics(output, original_tensor)
            quality_results[name] = metrics
            
            if verbose:
                print(f"📊 {name}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}, "
                      f"Edge={metrics['edge_preservation']:.3f}, Quality={metrics['quality_score']:.2f}")
        except Exception as e:
            if verbose:
                print(f"⚠️ {name}: 质量评估失败 - {str(e)}")
            # 为失败的输出分配低质量分数
            quality_results[name] = {
                'mse': float('inf'),
                'psnr': 0,
                'ssim': 0,
                'edge_preservation': 0,
                'color_diff': float('inf'),
                'quality_score': -1000
            }
    
    # 选择质量分数最高的输出
    best_name = max(quality_results.keys(), key=lambda x: quality_results[x]['quality_score'])
    best_output = candidates[best_name]
    
    # 为最佳输出选择最合适的mask
    best_mask = masks['mm']  # 默认使用模型生成的mask
    
    # 如果最佳输出不是最终精炼版本，可能需要调整mask策略
    if best_name != 'final_refined':
        if verbose:
            print(f"🎯 选择了非最终输出 {best_name}，保持原始mask策略")
    
    return best_output, best_mask, {
        'best_output': best_name,
        'all_metrics': quality_results,
        'best_metrics': quality_results[best_name]
    }


def create_ensemble_output(outputs, weights=None, verbose=False):
    """
    创建多输出的集成结果
    
    Args:
        outputs: dict - 包含所有模型输出的字典
        weights: dict - 每个输出的权重
        verbose: 是否打印详细信息
    
    Returns:
        torch.Tensor: 集成后的输出
    """
    if weights is None:
        # 默认权重：最终精炼输出权重最高
        weights = {
            'g_images': 0.5,    # 最终精炼输出
            'out3': 0.25,       # UNet输出
            'out2': 0.15,       # 尺度2输出
            'out1': 0.1         # 尺度1输出
        }
    
    # 使用最终精炼输出作为目标尺寸
    reference_output = outputs['g_images']
    if reference_output.dim() == 4:
        reference_output = reference_output.squeeze(0)
    target_size = reference_output.shape
    ensemble_result = torch.zeros_like(reference_output)
    
    total_weight = 0
    for output_name, weight in weights.items():
        if output_name in outputs:
            output = outputs[output_name]
            
            # 确保维度一致
            if output.dim() == 4:
                output = output.squeeze(0)
            
            # 调整尺寸到目标大小
            if output.shape != target_size:
                output = torch.nn.functional.interpolate(
                    output.unsqueeze(0), size=target_size[1:], mode='bilinear', align_corners=False
                ).squeeze(0)
            
            ensemble_result += output * weight
            total_weight += weight
            
            if verbose:
                print(f"📊 集成 {output_name}: 权重={weight:.2f}, 尺寸={output.shape}")
    
    # 归一化
    if total_weight > 0:
        ensemble_result /= total_weight
    
    return ensemble_result


def process_image(model_path, input_image_path, output_path, verbose=True, save_debug=False, extract_stamp=True, enhance_mask=True, use_grayscale_mask=True, use_ensemble=False, use_best_selection=True):
    """
    处理单张图像 - 印章擦除 + 多输出融合 + 基于灰度/二值化的印章提取
    
    Args:
        save_debug: 如果为True，保存调试输出（多尺度输出和原始模型mask）
        extract_stamp: 如果为True，提取印章和生成mask
        enhance_mask: 如果为True，进行mask质量改进
        use_grayscale_mask: 如果为True，使用灰度mask（避免锯齿）
        use_ensemble: 如果为True，使用集成输出
        use_best_selection: 如果为True，自动选择最佳输出
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"📱 使用设备: {device}")
    
    # 加载模型
    if verbose:
        print("🔧 加载模型...")
    model = STRnet2(3)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        if verbose:
            print(f"✅ 加载预训练模型: {os.path.basename(model_path)}")
    else:
        if verbose:
            print("⚠️  使用随机初始化模型（演示模式）")
    
    model.to(device)
    model.eval()
    
    # 处理图像
    if verbose:
        print(f"🖼️  处理图像: {os.path.basename(input_image_path)}")
    
    input_tensor = load_image(input_image_path).to(device)
    
    # 加载原始图像用于对比
    original_pil = Image.open(input_image_path).convert('RGB')
    
    with torch.no_grad():
        # 获取模型的所有输出
        out1, out2, out3, g_images, mm = model(input_tensor)
        
        # 转换到CPU并组织所有输出
        outputs = {
            'out1': out1.data.cpu(),
            'out2': out2.data.cpu(),
            'out3': out3.data.cpu(),
            'g_images': g_images.data.cpu(),
        }
        
        masks = {
            'mm': mm.data.cpu()
        }
        
        # 选择最佳输出策略
        if use_ensemble:
            # 使用集成输出
            if verbose:
                print("🔄 使用多输出集成策略")
            result = create_ensemble_output(outputs, verbose=verbose)
            final_mask = masks['mm']
            selection_stats = {'method': 'ensemble'}
            
        elif use_best_selection:
            # 自动选择最佳输出
            if verbose:
                print("🎯 自动选择最佳输出")
            result, final_mask, selection_stats = select_best_output_combination(
                outputs, input_tensor.cpu(), masks, verbose=verbose
            )
            
        else:
            # 使用传统的最终精炼输出
            if verbose:
                print("📱 使用传统最终精炼输出")
            result = outputs['g_images']
            final_mask = masks['mm']
            selection_stats = {'method': 'traditional', 'best_output': 'final_refined'}
        
        # 转换为PIL图像用于印章提取
        # 确保result是正确的维度
        if result.dim() == 4:
            result = result.squeeze(0)
        cleaned_pil = tensor_to_pil(result)
        
        # 调试模式：保存所有输出和质量对比
        if save_debug:
            # 保存所有模型输出
            for output_name, output_tensor in outputs.items():
                debug_path = output_path.replace('.jpg', f'_debug_{output_name}.jpg')
                ensure_dir(debug_path)
                save_result(output_tensor, debug_path)
            
            # 保存质量对比报告
            if use_best_selection and 'all_metrics' in selection_stats:
                quality_report_path = output_path.replace('.jpg', '_quality_report.txt')
                ensure_dir(quality_report_path)
                
                with open(quality_report_path, 'w', encoding='utf-8') as f:
                    f.write("=== 多输出质量对比报告 ===\n\n")
                    f.write(f"选择的最佳输出: {selection_stats['best_output']}\n\n")
                    
                    for name, metrics in selection_stats['all_metrics'].items():
                        f.write(f"{name}:\n")
                        f.write(f"  PSNR: {metrics['psnr']:.2f}\n")
                        f.write(f"  SSIM: {metrics['ssim']:.3f}\n")
                        f.write(f"  边缘保持: {metrics['edge_preservation']:.3f}\n")
                        f.write(f"  颜色差异: {metrics['color_diff']:.2f}\n")
                        f.write(f"  综合质量分: {metrics['quality_score']:.2f}\n\n")
            
            if verbose:
                print(f"🔧 调试输出已保存 (包含所有模型输出)")
                if 'best_output' in selection_stats:
                    print(f"🎯 最佳输出: {selection_stats['best_output']}")
        
        # 提取印章和生成mask
        if extract_stamp:
            if verbose:
                mode_str = "灰度mask" if use_grayscale_mask else "二值化mask"
                print(f"🔍 提取印章 ({mode_str})")
            
            # 使用改进的印章提取算法（使用选择的最佳mask）
            # 确保final_mask维度正确
            if final_mask.dim() == 4:
                final_mask = final_mask.squeeze(0)
            
            stamp_image, mask_image, diff_stats = extract_stamp_and_mask(
                original_pil, cleaned_pil, final_mask, 
                enhance_mask=enhance_mask,
                use_grayscale=use_grayscale_mask
            )
            
            # 添加选择策略信息到统计中
            diff_stats.update({
                'selection_method': selection_stats.get('method', 'unknown'),
                'selected_output': selection_stats.get('best_output', 'unknown')
            })
            
            # 保存必要输出：印章和mask
            stamp_path = output_path.replace('.jpg', '_stamp.jpg')
            mask_path = output_path.replace('.jpg', '_mask.png')
            
            ensure_dir(stamp_path)
            ensure_dir(mask_path)
            
            stamp_image.save(stamp_path)
            mask_image.save(mask_path)
            
            # 调试模式：保存原始模型mask和最终使用的mask
            if save_debug:
                # 保存原始模型mask
                model_mask_path = output_path.replace('.jpg', '_debug_model_mask.png')
                ensure_dir(model_mask_path)
                model_mask_vis = tensor_to_pil(masks['mm'])
                model_mask_vis.save(model_mask_path)
                
                # 总是保存最终使用的mask（即使相同也保存，便于调试对比）
                final_mask_path = output_path.replace('.jpg', '_debug_final_mask.png')
                ensure_dir(final_mask_path)
                final_mask_vis = tensor_to_pil(final_mask)
                final_mask_vis.save(final_mask_path)
                
                if verbose:
                    if torch.equal(final_mask, masks['mm']):
                        print(f"🔧 最终mask与模型mask相同")
                    else:
                        print(f"🔧 最终mask与模型mask不同")
            
            if verbose:
                print(f"📄 印章保存到: {os.path.basename(stamp_path)}")
                print(f"🎭 Mask保存到: {os.path.basename(mask_path)}")
                print(f"📊 印章区域占比: {diff_stats['stamp_ratio']:.2%}")
                print(f"📊 印章区域LAB差异: {diff_stats['stamp_lab_diff']:.2f}")
                print(f"📊 整体LAB差异: {diff_stats['mean_lab_diff']:.2f}")
                print(f"🎯 使用输出: {diff_stats.get('selected_output', 'unknown')}")
                print(f"🔄 选择方法: {diff_stats.get('selection_method', 'unknown')}")
                if save_debug:
                    print(f"🔧 调试文件已保存 (包含质量对比报告)")
    
    # 保存最终结果
    ensure_dir(output_path)
    save_result(result, output_path)
    
    if verbose:
        print(f"✅ 完成! 保存到: {output_path}")
        if save_debug:
            print("🔧 调试模式: 已保存额外的调试文件")
    
    return output_path


def batch_process(model_path, input_dir, output_dir, extract_stamp=True, use_ensemble=False, use_best_selection=True, save_debug_masks=False):
    """批量处理图像，包含基于多输出融合的印章提取"""
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')
    
    # 获取图像文件
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(supported_formats):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"❌ 在目录 {input_dir} 中未找到图像文件")
        return
    
    print(f"📁 找到 {len(image_files)} 个图像文件")
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    success_count = 0
    for i, image_path in enumerate(image_files):
        print(f"📊 进度: {i+1}/{len(image_files)} - {os.path.basename(image_path)}")
        try:
            output_filename = f"cleaned_{os.path.basename(image_path).replace('.png', '.jpg')}"
            output_path = os.path.join(output_dir, output_filename)
            process_image(
                model_path=model_path, 
                input_image_path=image_path, 
                output_path=output_path, 
                verbose=False,
                save_debug=save_debug_masks,
                extract_stamp=extract_stamp,
                use_ensemble=use_ensemble,
                use_best_selection=use_best_selection
            )
            success_count += 1
        except Exception as e:
            print(f"❌ 处理 {os.path.basename(image_path)} 出错: {str(e)}")
    
    print(f"🎉 批量处理完成! 成功处理 {success_count}/{len(image_files)} 个文件")
    print(f"📁 结果保存在: {output_dir}")
    if extract_stamp:
        print("📄 每个文件生成: 清理图像 + 印章图像 + Mask图像")


def quick_demo():
    """快速演示模式 - 多输出融合版本"""
    print("🚀 快速演示模式 (多输出融合)")
    print("-" * 50)
    
    # 创建演示图像
    demo_image = create_demo_image()
    demo_path = "demo_input.jpg"
    demo_image.save(demo_path)
    print(f"📝 创建演示图像: {demo_path}")
    
    # 处理图像 - 多输出融合模式
    output_path = "demo_output_final.jpg"
    process_image("./models/pre_model.pth", demo_path, output_path, save_debug=True, use_best_selection=True)
    
    print(f"\n🎉 演示完成!")
    print(f"   输入图像: {demo_path}")
    print(f"   清理图像: {output_path}")
    print(f"   印章图像: demo_output_final_stamp.jpg")
    print(f"   Mask图像: demo_output_final_mask.png")
    print("\n🔧 算法说明:")
    print("   - 充分利用模型的5个输出：out1, out2, out3, g_images, mm")
    print("   - 自动质量评估：PSNR, SSIM, 边缘保持度, 颜色一致性")
    print("   - 智能输出选择：自动选择质量最佳的模型输出")
    print("   - 多输出融合：可选的加权集成策略")
    print("   - 增强印章提取：基于最佳输出的印章分离")
    
    return demo_path, output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='印章擦除系统 (多输出融合印章提取)')
    parser.add_argument('--model_path', type=str, default='./models/pre_model.pth',
                        help='模型路径')
    parser.add_argument('--input_image', type=str, default=r'image\2.png', help='输入图像路径')
    parser.add_argument('--input_dir', type=str, help='输入图像目录（批量处理）')
    parser.add_argument('--output_path', type=str, default='./results/cleaned_image.jpg',
                        help='输出图像路径')
    parser.add_argument('--output_dir', type=str, default='./results/',
                        help='输出目录（批量处理）')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式：保存额外的调试文件')
    parser.add_argument('--no_extract', action='store_true',
                        help='不提取印章和mask')
    parser.add_argument('--no_enhance_mask', action='store_true',
                        help='不进行mask质量改进后处理')
    parser.add_argument('--binary_mask', action='store_true',
                        help='使用二值化mask（默认使用灰度mask避免锯齿）')
    parser.add_argument('--ensemble', action='store_true',
                        help='使用多输出集成策略（加权融合所有输出）')
    parser.add_argument('--no_auto_select', action='store_true',
                        help='禁用自动最佳输出选择（使用传统最终输出）')
    parser.add_argument('--batch_debug_masks', action='store_true',
                        help='批量处理时保存调试mask文件')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🎯 印章擦除系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    print()
    
    try:
        if args.input_image:
            # 单张图像处理
            if not os.path.exists(args.input_image):
                print(f"❌ 图像文件不存在: {args.input_image}")
                return
            mode_desc = "二值化" if args.binary_mask else "灰度"
            strategy_desc = "集成" if args.ensemble else ("智能选择" if not args.no_auto_select else "传统")
            print(f"📷 单张图像处理模式 ({mode_desc}mask, {strategy_desc}策略)")
            process_image(
                model_path=args.model_path, 
                input_image_path=args.input_image, 
                output_path=args.output_path, 
                save_debug=args.debug,
                extract_stamp=not args.no_extract,
                enhance_mask=not args.no_enhance_mask,
                use_grayscale_mask=not args.binary_mask,
                use_ensemble=args.ensemble,
                use_best_selection=not args.no_auto_select
            )
            
        elif args.input_dir:
            # 批量处理
            if not os.path.exists(args.input_dir):
                print(f"❌ 目录不存在: {args.input_dir}")
                return
            strategy_desc = "集成" if args.ensemble else ("智能选择" if not args.no_auto_select else "传统")
            print(f"📁 批量处理模式 ({strategy_desc}策略印章提取)")
            batch_process(args.model_path, args.input_dir, args.output_dir, 
                         extract_stamp=not args.no_extract,
                         use_ensemble=args.ensemble,
                         use_best_selection=not args.no_auto_select,
                         save_debug_masks=args.batch_debug_masks)
            
        else:
            # 默认演示模式
            quick_demo()
            print("\n💡 使用说明:")
            print("  单张处理: python example.py --input_image image.jpg")
            print("  批量处理: python example.py --input_dir images/ --output_dir results/")
            print("  批量调试mask: python example.py --input_dir images/ --batch_debug_masks")
            print("  集成策略: python example.py --input_image image.jpg --ensemble")
            print("  传统模式: python example.py --input_image image.jpg --no_auto_select")
            print("  调试模式: python example.py --input_image image.jpg --debug")
            print("  二值化mask: python example.py --input_image image.jpg --binary_mask")
            
    except Exception as e:
        print(f"❌ 运行出错: {str(e)}")
        print("请检查模型文件和依赖是否正确安装")


if __name__ == "__main__":
    main()
