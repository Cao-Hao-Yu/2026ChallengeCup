# import os
# import cv2

# # ================= 配置路径 =================
# IMAGE_DIR = r"D:\DeepLearning\Challenger\data\dataset_yolo\images\train"

# def get_area_range_name(area):
#     """
#     根据面积返回区间名称
#     """
#     # 区间定义 (单位：像素)
#     if area < 100 * 100:
#         return "< 100x100"
#     elif area < 200 * 200:
#         return "100x100 - 200x200"
#     elif area < 300 * 300:
#         return "200x200 - 300x300"
#     elif area < 400 * 400:
#         return "300x300 - 400x400"
#     elif area < 500 * 500:
#         return "400x400 - 500x500"
#     elif area < 600 * 600:
#         return "500x500 - 600x600"
#     elif area < 800 * 800:
#         return "600x600 - 800x800"
#     elif area < 1000 * 1000:
#         return "800x800 - 1000x1000"
#     elif area < 2000 * 2000:
#         return "1K x 1K - 2K x 2K"
#     elif area < 4000 * 4000:
#         return "2K x 2K - 4K x 4K"
#     else:
#         return "> 4K x 4K"

# def main():
#     if not os.path.exists(IMAGE_DIR):
#         print(f"错误：路径不存在 - {IMAGE_DIR}")
#         return

#     valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
#     image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
#     print(f"开始分析，共发现 {len(image_files)} 张图片...")

#     # 初始化统计字典
#     stats = {
#         "< 100x100": 0,
#         "100x100 - 200x200": 0,
#         "200x200 - 300x300": 0,
#         "300x300 - 400x400": 0,
#         "400x400 - 500x500": 0,
#         "500x500 - 600x600": 0,
#         "600x600 - 800x800": 0,
#         "800x800 - 1000x1000": 0,
#         "1K x 1K - 2K x 2K": 0,
#         "2K x 2K - 4K x 4K": 0,
#         "> 4K x 4K": 0
#     }

#     error_count = 0
#     total_processed = 0

#     for filename in image_files:
#         file_path = os.path.join(IMAGE_DIR, filename)
        
#         try:
#             img = cv2.imread(file_path)
#             if img is None:
#                 error_count += 1
#                 continue
            
#             h, w = img.shape[:2]
#             area = w * h
            
#             # 获取区间名称并计数
#             range_name = get_area_range_name(area)
#             if range_name in stats:
#                 stats[range_name] += 1
#             else:
#                 # 如果有超出预设区间的，归入最大类或打印警告
#                 print(f"发现超大/超小尺寸: {w}x{h}, 面积: {area}")
#                 stats["> 4K x 4K"] += 1 
            
#             total_processed += 1
            
#         except Exception as e:
#             print(f"处理 {filename} 出错: {e}")
#             error_count += 1

#     # ================= 输出结果 =================
#     print("\n" + "="*40)
#     print(f"图片尺寸面积分布统计 (总有效图片: {total_processed})")
#     print("="*40)
    
#     # 遍历字典输出
#     for range_name, count in stats.items():
#         if count > 0:
#             percentage = (count / total_processed) * 100
#             # 为了对齐，格式化输出
#             print(f"{range_name:<20} : {count:>5} 张 ({percentage:>5.2f}%)")

#     if error_count > 0:
#         print(f"\n警告: 有 {error_count} 张图片损坏或读取失败。")

# if __name__ == "__main__":
#     main()


import os

def count_files_without_mar(folder_path):
    """
    统计指定文件夹中不包含'MAR'字段的文件数量
    
    参数:
        folder_path (str): 要检查的文件夹路径
        
    返回:
        int: 不包含'MAR'字段的文件数量
    """
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if 'MAR' not in file:
                count += 1
    return count

# 使用示例
if __name__ == '__main__':
    folder_path = r"D:\DeepLearning\Challenger\data\raw_dataset\labels\train"
    if os.path.isdir(folder_path):
        result = count_files_without_mar(folder_path)
        print(f"在文件夹 '{folder_path}' 中，不包含'MAR'字段的文件共有: {result} 个")
    else:
        print("错误: 指定的路径不是一个有效的文件夹")
