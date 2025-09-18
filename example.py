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


def process_image(model_path, input_image_path, output_path, verbose=True, save_debug=False, extract_stamp=True, enhance_mask=True, use_grayscale_mask=True):
    """
    处理单张图像 - 印章擦除 + 基于灰度/二值化的印章提取
    
    Args:
        save_debug: 如果为True，保存调试输出（多尺度输出和原始模型mask）
        extract_stamp: 如果为True，提取印章和生成mask
        enhance_mask: 如果为True，进行mask质量改进
        use_grayscale_mask: 如果为True，使用灰度mask（避免锯齿）
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
        
        # 转换到CPU
        g_image = g_images.data.cpu()
        
        # 关键修复：在推理时，我们应该直接使用生成的图像
        # 原始test.py中的mask混合是用于训练/测试时有ground truth的情况
        # 在实际推理时，模型的最终输出g_images就是我们要的结果
        result = g_image
        
        # 转换为PIL图像用于印章提取
        cleaned_pil = tensor_to_pil(result)
        
        # 调试模式：保存多尺度输出
        if save_debug:
            out1_cpu = out1.data.cpu()
            out2_cpu = out2.data.cpu() 
            out3_cpu = out3.data.cpu()
            
            out1_path = output_path.replace('.jpg', '_debug_out1.jpg')
            out2_path = output_path.replace('.jpg', '_debug_out2.jpg')
            out3_path = output_path.replace('.jpg', '_debug_out3.jpg')
            
            ensure_dir(out1_path)
            save_result(out1_cpu, out1_path)
            save_result(out2_cpu, out2_path)
            save_result(out3_cpu, out3_path)
            
            if verbose:
                print(f"🔧 调试输出已保存")
        
        # 提取印章和生成mask
        if extract_stamp:
            if verbose:
                mode_str = "灰度mask" if use_grayscale_mask else "二值化mask"
                print(f"🔍 提取印章 ({mode_str})")
            
            # 使用印章提取算法
            stamp_image, mask_image, diff_stats = extract_stamp_and_mask(
                original_pil, cleaned_pil, mm, 
                enhance_mask=enhance_mask,
                use_grayscale=use_grayscale_mask
            )
            
            # 保存必要输出：印章和mask
            stamp_path = output_path.replace('.jpg', '_stamp.jpg')
            mask_path = output_path.replace('.jpg', '_mask.png')
            
            ensure_dir(stamp_path)
            ensure_dir(mask_path)
            
            stamp_image.save(stamp_path)
            mask_image.save(mask_path)
            
            # 调试模式：保存原始模型mask
            if save_debug:
                model_mask_path = output_path.replace('.jpg', '_debug_model_mask.png')
                ensure_dir(model_mask_path)
                model_mask_vis = tensor_to_pil(mm)
                model_mask_vis.save(model_mask_path)
            
            if verbose:
                print(f"📄 印章保存到: {os.path.basename(stamp_path)}")
                print(f"🎭 Mask保存到: {os.path.basename(mask_path)}")
                print(f"📊 印章区域占比: {diff_stats['stamp_ratio']:.2%}")
                print(f"📊 印章区域LAB差异: {diff_stats['stamp_lab_diff']:.2f}")
                print(f"📊 整体LAB差异: {diff_stats['mean_lab_diff']:.2f}")
                if save_debug:
                    print(f"🔧 调试mask保存到: {os.path.basename(model_mask_path)}")
    
    # 保存最终结果
    ensure_dir(output_path)
    save_result(result, output_path)
    
    if verbose:
        print(f"✅ 完成! 保存到: {output_path}")
        if save_debug:
            print("🔧 调试模式: 已保存额外的调试文件")
    
    return output_path


def batch_process(model_path, input_dir, output_dir, extract_stamp=True):
    """批量处理图像，包含基于模型mask的印章提取"""
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
                extract_stamp=extract_stamp
            )
            success_count += 1
        except Exception as e:
            print(f"❌ 处理 {os.path.basename(image_path)} 出错: {str(e)}")
    
    print(f"🎉 批量处理完成! 成功处理 {success_count}/{len(image_files)} 个文件")
    print(f"📁 结果保存在: {output_dir}")
    if extract_stamp:
        print("📄 每个文件生成: 清理图像 + 印章图像 + Mask图像")


def quick_demo():
    """快速演示模式 - OTSU二值化版本"""
    print("🚀 快速演示模式 (OTSU二值化)")
    print("-" * 50)
    
    # 创建演示图像
    demo_image = create_demo_image()
    demo_path = "demo_input.jpg"
    demo_image.save(demo_path)
    print(f"📝 创建演示图像: {demo_path}")
    
    # 处理图像 - 标准模式
    output_path = "demo_output_final.jpg"
    process_image("./models/pre_model.pth", demo_path, output_path, save_debug=False)
    
    print(f"\n🎉 演示完成!")
    print(f"   输入图像: {demo_path}")
    print(f"   清理图像: {output_path}")
    print(f"   印章图像: demo_output_final_stamp.jpg")
    print(f"   Mask图像: demo_output_final_mask.png")
    print("\n🔧 算法说明:")
    print("   - 修复了图像预处理（移除错误归一化）")
    print("   - 修复了推理逻辑（直接使用模型生成图像）")
    print("   - 使用OTSU算法自动二值化模型mask")
    print("   - 简单高效，避免过度后处理失真")
    
    return demo_path, output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='印章擦除系统 (OTSU二值化印章提取)')
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
            print(f"📷 单张图像处理模式 ({mode_desc}mask印章提取)")
            process_image(
                model_path=args.model_path, 
                input_image_path=args.input_image, 
                output_path=args.output_path, 
                save_debug=args.debug,
                extract_stamp=not args.no_extract,
                enhance_mask=not args.no_enhance_mask,
                use_grayscale_mask=not args.binary_mask
            )
            
        elif args.input_dir:
            # 批量处理
            if not os.path.exists(args.input_dir):
                print(f"❌ 目录不存在: {args.input_dir}")
                return
            print("📁 批量处理模式 (OTSU二值化印章提取)")
            batch_process(args.model_path, args.input_dir, args.output_dir, 
                         extract_stamp=not args.no_extract)
            
        else:
            # 默认演示模式
            quick_demo()
            print("\n💡 使用说明:")
            print("  单张处理: python example.py --input_image image.jpg")
            print("  批量处理: python example.py --input_dir images/ --output_dir results/")
            print("  不提取印章: python example.py --input_image image.jpg --no_extract")
            print("  调试模式: python example.py --input_image image.jpg --debug")
            print("  二值化mask: python example.py --input_image image.jpg --binary_mask")
            
    except Exception as e:
        print(f"❌ 运行出错: {str(e)}")
        print("请检查模型文件和依赖是否正确安装")


if __name__ == "__main__":
    main()
