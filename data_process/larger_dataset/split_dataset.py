"""划分数据集"""
import os
import shutil
import random
from collections import defaultdict

def split_yolo_dataset(
    src_img_dir, 
    src_lbl_dir, 
    dst_train_img_dir, 
    dst_train_lbl_dir, 
    dst_val_img_dir, 
    dst_val_lbl_dir, 
    split_ratio=0.2, 
    split_strategy='stratified',
    seed=42
):
    """
    划分 YOLO 格式数据集为训练集和验证集
    
    Args:
        src_img_dir (str): 原始图像路径
        src_lbl_dir (str): 原始标签路径
        dst_train_img_dir (str): 划分后的训练图像路径
        dst_train_lbl_dir (str): 划分后的训练标签路径
        dst_val_img_dir (str): 划分后的验证图像路径
        dst_val_lbl_dir (str): 划分后的验证标签路径
        split_ratio (float): 验证集比例，默认 0.2
        split_strategy (str): 划分策略，'stratified' (分层) 或 'random' (随机)
        seed (int): 随机种子
    """
    
    random.seed(seed)
    
    # 创建目标目录
    for dir_path in [dst_train_img_dir, dst_train_lbl_dir, dst_val_img_dir, dst_val_lbl_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print("正在扫描数据集...")
    file_info = []
    missing_labels = 0
    
    for img_filename in os.listdir(src_img_dir):
        if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
            continue
        
        img_path = os.path.join(src_img_dir, img_filename)
        lbl_filename = os.path.splitext(img_filename)[0] + '.txt'
        lbl_path = os.path.join(src_lbl_dir, lbl_filename)
        
        # 检查标签是否存在
        if not os.path.exists(lbl_path):
            missing_labels += 1
            continue
            
        # 读取文件中的类别信息
        classes_in_file = set()
        try:
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes_in_file.add(int(parts[0]))
        except Exception as e:
            print(f"读取标签文件错误 {lbl_filename}: {e}")
            continue

        # 确定主类别（用于分层抽样），取最小的ID
        main_class = min(classes_in_file) if classes_in_file else -1
        
        file_info.append({
            'img_filename': img_filename,
            'lbl_filename': lbl_filename,
            'img_path': img_path,
            'lbl_path': lbl_path,
            'main_class': main_class,
            'classes': classes_in_file
        })
    
    print(f"扫描完成。共发现 {len(file_info)} 对有效文件。")
    if missing_labels > 0:
        print(f"警告：发现 {missing_labels} 个图片缺少对应的标签文件，已跳过。")

    train_files = []
    val_files = []

    # ==========================================================
    # 划分逻辑选择
    # ==========================================================
    if split_strategy == 'random':
        # --- 模式 1: 普通随机划分 ---
        print("使用普通随机划分模式...")
        random.shuffle(file_info)
        split_idx = int(len(file_info) * (1 - split_ratio))
        
        train_files = file_info[:split_idx]
        val_files = file_info[split_idx:]

    elif split_strategy == 'stratified':
        # --- 模式 2: 分层抽样 (保证每个类别都有) ---
        print("使用分层抽样模式 (Stratified Split)...")
        files_by_class = defaultdict(list)
        for info in file_info:
            files_by_class[info['main_class']].append(info)
        
        print(f"检测到 {len(files_by_class)} 个主类别。")
        
        for cls_id, files in files_by_class.items():
            random.shuffle(files)
            val_count = int(len(files) * split_ratio)
            
            val_files.extend(files[:val_count])
            train_files.extend(files[val_count:])

    else:
        raise ValueError(f"未知的划分策略: {split_strategy}。请选择 'random' 或 'stratified'。")

    # ==========================================================
    # 结果统计与文件复制
    # ==========================================================
    print(f"\n划分结果: Train: {len(train_files)} 张, Val: {len(val_files)} 张")
    if len(file_info) > 0:
        print(f"验证集比例: {len(val_files) / len(file_info) * 100:.2f}%")

    # 如果是分层模式，输出一下验证集类别分布检查
    if split_strategy == 'stratified':
        val_class_dist = defaultdict(int)
        for info in val_files:
            for cls in info['classes']:
                val_class_dist[cls] += 1
        print("\n验证集各类别实例数量分布 (前10个):")
        for cls_id in sorted(val_class_dist.keys())[:10]:
            print(f"  类别 {cls_id}: {val_class_dist[cls_id]} 个实例")

    # 开始复制文件
    print("\n开始复制文件到目标目录...")
    process_split(train_files, dst_train_img_dir, dst_train_lbl_dir, "Train")
    process_split(val_files, dst_val_img_dir, dst_val_lbl_dir, "Val")
    
    print("数据集划分完成！")

def process_split(file_list, target_img_dir, target_lbl_dir, split_name):
    """辅助函数：执行文件复制操作"""
    count = 0
    for info in file_list:
        try:
            # 复制图片
            shutil.copy2(info['img_path'], os.path.join(target_img_dir, info['img_filename']))
            # 复制标签
            shutil.copy2(info['lbl_path'], os.path.join(target_lbl_dir, info['lbl_filename']))
            count += 1
        except Exception as e:
            print(f"复制文件出错 {info['img_filename']}: {e}")
    
    print(f"已复制 {count} 个文件到 {split_name} 目录。")


if __name__ == "__main__":
    # ================= 配置区域 =================
    source_image_dir = r""
    source_label_dir = r""

    # 目标路径
    target_train_img = r""
    target_train_lbl = r""
    target_val_img   = r""
    target_val_lbl   = r""
    
    # ===========================================

    # 调用函数
    split_yolo_dataset(
        src_img_dir=source_image_dir,
        src_lbl_dir=source_label_dir,
        dst_train_img_dir=target_train_img,
        dst_train_lbl_dir=target_train_lbl,
        dst_val_img_dir=target_val_img,
        dst_val_lbl_dir=target_val_lbl,
        split_ratio=0.2,
        split_strategy='stratified',
        seed=42
    )
