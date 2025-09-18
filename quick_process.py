#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速处理指定样本 - 简化版本
"""

import os
from example import process_image

def main():
    """快速处理样本 - 包含印章提取"""
    print("🚀 快速样本处理 (印章擦除 + 印章提取)")
    
    # 样本文件列表
    samples = [
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\102_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\103_extracted_RGB.png", 
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\10_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\2_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\3_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\4_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\52_extracted_RGB.png",
        r"I:\pdfai_serv\classify\seal_extract\tmp_work\separated\6_extracted_RGB.png"
    ]
    
    model_path = "./models/pre_model.pth"
    output_dir = "./quick_results_with_stamps"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 处理 {len(samples)} 个样本...")
    print("-" * 50)
    
    success_count = 0
    for i, input_path in enumerate(samples, 1):
        if not os.path.exists(input_path):
            print(f"{i:2d}. ❌ 文件不存在: {os.path.basename(input_path)}")
            continue
        
        try:
            filename = os.path.basename(input_path).replace('.png', '.jpg')
            output_path = os.path.join(output_dir, f"cleaned_{filename}")
            
            # 使用基于模型mask的印章提取
            process_image(
                model_path=model_path, 
                input_image_path=input_path, 
                output_path=output_path, 
                verbose=False,
                extract_stamp=True  # 提取印章
            )
            
            print(f"{i:2d}. ✅ {os.path.basename(input_path)}")
            print(f"     📄 清理图像: cleaned_{filename}")
            print(f"     🔴 印章图像: cleaned_{filename.replace('.jpg', '_stamp.jpg')}")
            print(f"     🎭 Mask图像: cleaned_{filename.replace('.jpg', '_mask.png')}")
            success_count += 1
            
        except Exception as e:
            print(f"{i:2d}. ❌ {os.path.basename(input_path)}: {str(e)}")
    
    print("-" * 50)
    print(f"🎉 处理完成! 成功处理 {success_count}/{len(samples)} 个样本")
    print(f"📁 结果保存在: {output_dir}")
    print("\n📊 每个样本生成3个必要文件:")
    print("   - cleaned_*.jpg: 印章擦除后的清理图像")
    print("   - *_stamp.jpg: 提取的印章图像(白色背景)")
    print("   - *_mask.png: 印章区域mask(黑色=印章，白色=背景)")
    print("\n🔧 算法说明: 使用OTSU算法自动二值化模型mask，简单高效")
    print("💡 提示: 使用 --debug 参数可保存额外的调试文件")

if __name__ == "__main__":
    main()
