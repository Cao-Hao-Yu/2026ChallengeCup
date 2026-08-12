import sys
import os
from PIL import Image

def batch_convert_to_gray(input_dir, output_dir):
    """
    批量将文件夹下的RGB图像转换为灰度图
    
    参数:
    input_dir (str): 原始图像文件夹路径
    output_dir (str): 输出图像文件夹路径
    """
    # 1. 检查输入文件夹是否存在
    if not os.path.isdir(input_dir):
        print(f"错误：输入路径 '{input_dir}' 不是一个有效的文件夹。")
        return

    # 2. 处理输出文件夹逻辑
    # 如果输出文件夹和输入文件夹不同，且不存在，则创建它
    if input_dir != output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 3. 定义支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    print(f"正在处理文件夹: {input_dir}")
    count = 0
    
    try:
        # 遍历文件夹
        for filename in os.listdir(input_dir):
            # 检查是否为图片文件 (忽略大小写)
            if filename.lower().endswith(valid_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                
                try:
                    # 打开图片
                    with Image.open(input_path) as img:
                        # 转换为灰度图 ('L' 模式)
                        gray_img = img.convert('L')
                        
                        # 保存图片 (如果是同路径，这步就是覆盖)
                        gray_img.save(output_path)
                        count += 1
                        print(f"已转换: {filename}")
                        
                except Exception as e:
                    print(f"警告：无法处理文件 {filename}，原因: {e}")
                    
    except Exception as e:
        print(f"发生错误: {e}")
    
    print("-" * 30)
    print(f"处理完成。共转换 {count} 张图片。")
    if input_dir == output_dir:
        print("注意：原图已被灰度图覆盖。")

if __name__ == "__main__":
    input_folder = r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\images"
    output_folder = r"D:\DeepLearning\Challenger\data\LargerDataset\slice_image_check"
    batch_convert_to_gray(input_folder, output_folder)