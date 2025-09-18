#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理指定样本的印章擦除脚本
复用 example.py 的函数来处理特定的样本文件

使用方法:
python process_samples.py
"""

import os
import sys
from pathlib import Path

# 导入 example.py 中的函数
from example import process_image, check_dependencies

def main():
    """处理指定的样本文件"""
    print("=" * 60)
    print("🎯 印章擦除 - 样本处理脚本")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 定义样本文件路径
    sample_files = [
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\102_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\103_extracted_RGB.png", 
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\10_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\2_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\3_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\4_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\52_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\6_extracted_RGB.png"
    ]
    
    # 配置参数
    model_path = "./models/pre_model.pth"
    output_dir = "./sample_results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 输出目录: {output_dir}")
    print(f"🔧 模型路径: {model_path}")
    print(f"📊 待处理样本数量: {len(sample_files)}")
    print("-" * 60)
    
    # 检查文件存在性
    existing_files = []
    missing_files = []
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  以下文件不存在:")
        for file_path in missing_files:
            print(f"   ❌ {file_path}")
        print()
    
    if not existing_files:
        print("❌ 没有找到任何有效的样本文件!")
        return
    
    print(f"✅ 找到 {len(existing_files)} 个有效样本文件")
    print()
    
    # 处理每个样本
    success_count = 0
    error_count = 0
    
    for i, input_path in enumerate(existing_files, 1):
        print(f"📷 处理样本 {i}/{len(existing_files)}")
        print(f"   输入: {os.path.basename(input_path)}")
        
        try:
            # 生成输出文件名
            input_filename = Path(input_path).stem  # 不包含扩展名
            output_filename = f"cleaned_{input_filename}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # 处理图像 (带调试mask模式)
            result_path = process_image(
                model_path=model_path,
                input_image_path=input_path,
                output_path=output_path,
                verbose=False,  # 减少输出信息
                save_debug=True,  # 保存调试文件，包括调试mask
                extract_stamp=True  # 使用优化的印章提取算法
            )
            
            print(f"   ✅ 成功: {os.path.basename(output_path)}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ 失败: {str(e)}")
            error_count += 1
        
        print()
    
    # 输出处理结果统计
    print("=" * 60)
    print("📊 处理结果统计:")
    print(f"   ✅ 成功处理: {success_count} 个文件")
    print(f"   ❌ 处理失败: {error_count} 个文件")
    print(f"   📁 结果保存在: {output_dir}")
    
    if success_count > 0:
        print("\n💡 输出文件说明:")
        print("   - cleaned_*.jpg: 印章擦除结果")
        print("   - *_stamp.jpg: 基于模型mask提取的印章")
        print("   - *_mask.png: 优化后的印章区域mask")
        print("   - *_debug_model_mask.png: 原始模型mask")
        print("   - *_debug_final_mask.png: 最终使用的mask")
        print("   - *_debug_*.jpg: 所有模型输出")
        print("   - *_quality_report.txt: 质量对比报告")
        print("\n🔧 注意: 已启用调试模式，保存所有调试文件")
    
    print("=" * 60)


def batch_process_with_details():
    """带详细信息的批量处理函数 (为每个样本创建子目录)"""
    print("🔍 详细处理模式 (子目录模式)")
    print("-" * 40)
    
    # 样本文件信息
    samples_info = [
        {"path": r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\102_extracted_RGB.png", "desc": "样本102"},
        {"path": r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\103_extracted_RGB.png", "desc": "样本103"},
        {"path": r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\10_extracted_RGB.png", "desc": "样本10"},
        {"path": r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\2_extracted_RGB.png", "desc": "样本2"},
        {"path": r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\3_extracted_RGB.png", "desc": "样本3"},
        {"path": r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\4_extracted_RGB.png", "desc": "样本4"},
        {"path": r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\52_extracted_RGB.png", "desc": "样本52"},
        {"path": r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\6_extracted_RGB.png", "desc": "样本6"}
    ]
    
    model_path = "./models/pre_model.pth"
    output_dir = "./detailed_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, sample in enumerate(samples_info, 1):
        input_path = sample["path"]
        desc = sample["desc"]
        
        if not os.path.exists(input_path):
            print(f"⚠️  {desc}: 文件不存在")
            continue
        
        print(f"\n🖼️  处理 {desc} ({i}/{len(samples_info)})")
        print(f"   文件: {os.path.basename(input_path)}")
        
        try:
            # 为每个样本创建子目录
            sample_output_dir = os.path.join(output_dir, f"sample_{Path(input_path).stem}")
            os.makedirs(sample_output_dir, exist_ok=True)
            
            output_path = os.path.join(sample_output_dir, "result.jpg")
            
            # 详细处理
            process_image(
                model_path=model_path,
                input_image_path=input_path,
                output_path=output_path,
                verbose=True,
                save_debug=True
            )
            
            print(f"   📁 结果保存在: {sample_output_dir}")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {str(e)}")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        batch_process_with_details()
    else:
        main()
