#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速处理指定样本 - 简化版本
"""

import os
from example import process_image

def main():
    """快速处理样本"""
    print("🚀 快速样本处理")
    
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
    output_dir = "./quick_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 处理 {len(samples)} 个样本...")
    print("-" * 40)
    
    for i, input_path in enumerate(samples, 1):
        if not os.path.exists(input_path):
            print(f"{i:2d}. ❌ 文件不存在: {os.path.basename(input_path)}")
            continue
        
        try:
            filename = os.path.basename(input_path).replace('.png', '.jpg')
            output_path = os.path.join(output_dir, f"cleaned_{filename}")
            
            process_image(model_path, input_path, output_path, verbose=False)
            print(f"{i:2d}. ✅ {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"{i:2d}. ❌ {os.path.basename(input_path)}: {str(e)}")
    
    print("-" * 40)
    print(f"🎉 处理完成! 结果在: {output_dir}")

if __name__ == "__main__":
    main()
