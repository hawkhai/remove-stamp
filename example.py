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
    
    if not deps_ok:
        print("\n❌ 请先安装必要的依赖:")
        print("pip install torch torchvision pillow numpy")
    
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


def process_image(model_path, input_image_path, output_path, verbose=True, save_both=False):
    """
    处理单张图像 - 修复：正确处理模型输出
    
    Args:
        save_both: 如果为True，保存两个版本：直接输出和mask混合版本
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
    
    with torch.no_grad():
        # 获取模型的所有输出
        out1, out2, out3, g_images, mm = model(input_tensor)
        
        # 转换到CPU
        g_image = g_images.data.cpu()
        
        # 关键修复：在推理时，我们应该直接使用生成的图像
        # 原始test.py中的mask混合是用于训练/测试时有ground truth的情况
        # 在实际推理时，模型的最终输出g_images就是我们要的结果
        result = g_image
        
        if save_both:
            # 保存其他尺度的输出用于调试
            out1_cpu = out1.data.cpu()
            out2_cpu = out2.data.cpu() 
            out3_cpu = out3.data.cpu()
            
            save_result(out1_cpu, output_path.replace('.jpg', '_out1.jpg'))
            save_result(out2_cpu, output_path.replace('.jpg', '_out2.jpg'))
            save_result(out3_cpu, output_path.replace('.jpg', '_out3.jpg'))
            
            if verbose:
                print(f"💾 多尺度输出已保存用于调试")
    
    # 保存最终结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_result(result, output_path)
    
    if verbose:
        print(f"✅ 完成! 保存到: {output_path}")
        if save_both:
            print("📝 说明: 保存了主输出和多尺度调试输出")
    
    return output_path


def batch_process(model_path, input_dir, output_dir):
    """批量处理图像"""
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
    for i, image_path in enumerate(image_files):
        print(f"📊 进度: {i+1}/{len(image_files)}")
        try:
            output_filename = f"cleaned_{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            process_image(model_path, image_path, output_path, verbose=False)
        except Exception as e:
            print(f"❌ 处理 {os.path.basename(image_path)} 出错: {str(e)}")
    
    print(f"🎉 批量处理完成! 结果保存在: {output_dir}")


def quick_demo():
    """快速演示模式 - 最终修复版本"""
    print("🚀 快速演示模式 (最终修复版本)")
    print("-" * 50)
    
    # 创建演示图像
    demo_image = create_demo_image()
    demo_path = "demo_input.jpg"
    demo_image.save(demo_path)
    print(f"📝 创建演示图像: {demo_path}")
    
    # 处理图像 - 保存多尺度输出用于调试
    output_path = "demo_output_final.jpg"
    process_image("./models/pre_model.pth", demo_path, output_path, save_both=True)
    
    print(f"\n🎉 演示完成!")
    print(f"   输入图像: {demo_path}")
    print(f"   主要输出: {output_path}")
    print(f"   调试输出: demo_output_final_out1/2/3.jpg")
    print("\n🔧 最终修复说明:")
    print("   - 修复了图像预处理（移除错误归一化）")
    print("   - 修复了推理逻辑（直接使用模型生成图像）")
    print("   - 移除了错误的mask混合（推理时不需要）")
    print("   - 现在直接输出模型的印章擦除结果")
    
    return demo_path, output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='印章擦除系统 (已修复推理逻辑)')
    parser.add_argument('--model_path', type=str, default='./models/pre_model.pth',
                        help='模型路径')
    parser.add_argument('--input_image', type=str, default=r'image\2.png', help='输入图像路径')
    parser.add_argument('--input_dir', type=str, help='输入图像目录（批量处理）')
    parser.add_argument('--output_path', type=str, default='./results/cleaned_image.jpg',
                        help='输出图像路径')
    parser.add_argument('--output_dir', type=str, default='./results/',
                        help='输出目录（批量处理）')
    parser.add_argument('--save_both', action='store_true',
                        help='保存两个版本：mask混合版本和直接输出版本')
    
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
            print("📷 单张图像处理模式 (已修复)")
            process_image(args.model_path, args.input_image, args.output_path, save_both=args.save_both)
            
        elif args.input_dir:
            # 批量处理
            if not os.path.exists(args.input_dir):
                print(f"❌ 目录不存在: {args.input_dir}")
                return
            print("📁 批量处理模式 (已修复)")
            batch_process(args.model_path, args.input_dir, args.output_dir)
            
        else:
            # 默认演示模式
            quick_demo()
            print("\n💡 使用说明:")
            print("  单张处理: python example.py --input_image image.jpg")
            print("  批量处理: python example.py --input_dir images/ --output_dir results/")
            print("  保存两版本: python example.py --input_image image.jpg --save_both")
            
    except Exception as e:
        print(f"❌ 运行出错: {str(e)}")
        print("请检查模型文件和依赖是否正确安装")


if __name__ == "__main__":
    main()
