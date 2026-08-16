import os

# ================= 配置区域 =================

# HBB 标签文件所在路径
hbb_folder = r'D:\DeepLearning\Challenger\data\LargerDataset\OurDataset\hbblabels'

# 类别映射字典
CLASS_MAP = {
    '0': '0:0',
    '1': '0:1',
    '2': '0:2',
    '3': '0:3',
    '4': '1:4',
    '5': '1:5',
    '6': '1:6',
    '7': '1:7',
    '8': '1:8',
    '9': '1:9',
    '10': '1:10',
    '11': '1:11',
    '12': '1:12',
    '13': '1:13',
    '14': '1:14',
    '15': '1:15',
    '16': '1:16',
    '17': '1:17',
    '18': '1:18',
    '19': '1:19',
    '20': '1:20',
    '21': '1:21',
    '22': '1:22',
    '23': '1:23',
    '24': '2:24',
}

IGNORE_CLASSES = {
}

# ===========================================

def check_unmapped_classes(folder_path, class_map, ignore_set):
    """
    扫描文件夹中的所有标签文件，找出既未映射也未忽略的类别ID
    """
    unmapped_ids = set()
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not files:
        print(f"警告: 在路径 {folder_path} 下没有找到 .txt 文件")
        return unmapped_ids
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        original_id = parts[0]
                        if original_id not in class_map and original_id not in ignore_set:
                            unmapped_ids.add(original_id)
        except Exception as e:
            print(f"读取文件 {filename} 出错: {e}")
            
    return unmapped_ids

def update_label_file(folder_path, class_map, ignore_set):
    """
    更新指定文件夹下的所有标签文件，保留层级字符串格式
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    updated_count = 0
    ignored_count = 0
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        new_lines = []
        needs_update = False
        file_ignored = 0
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                original_id = parts[0]
                
                # 逻辑判断
                if original_id in ignore_set:
                    # 如果在忽略列表中，跳过该行
                    file_ignored += 1
                    needs_update = True 
                    continue
                
                if original_id in class_map:
                    # 直接替换为映射后的字符串（保留冒号）
                    # 例如：parts[0] 变为 "0:1"
                    parts[0] = class_map[original_id]
                    needs_update = True
                
                # 即使不在映射表中（前面检查过应该都在），也按原样保留，防止数据丢失
                new_lines.append(' '.join(parts))
            
            # 只有当内容发生变化时才重写文件
            if needs_update:
                with open(filepath, 'w') as f:
                    f.write('\n'.join(new_lines))
                updated_count += 1
                ignored_count += file_ignored
                
        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")
            
    return updated_count, ignored_count

def main():
    print("=" * 50)
    print("开始执行类别映射检查与转换程序 (保留层级格式)")
    print("=" * 50)
    
    # 1. 检查路径
    if not os.path.exists(hbb_folder):
        print(f"[错误] 路径不存在: {hbb_folder}")
        return

    # 2. 检查未映射的类别
    print(f"\n[1/2] 正在检查路径: {hbb_folder}")
    unmapped = check_unmapped_classes(hbb_folder, CLASS_MAP, IGNORE_CLASSES)
    
    if unmapped:
        print("\n[错误] 发现未映射且未声明忽略的类别 ID！请补充 CLASS_MAP 配置：")
        for uid in sorted(list(unmapped)):
            print(f"    '{uid}': '0:0',  # <- 请确认此ID的层级")
        print("\n程序已暂停，请修改配置后重新运行。")
        return
    else:
        print("[检查通过] 所有类别 ID 均已在配置中定义。")
    
    # 3. 执行更新
    print(f"\n[2/2] 正在更新标签文件...")
    files_count, objs_count = update_label_file(hbb_folder, CLASS_MAP, IGNORE_CLASSES)
    print(f"  -> 已更新 {files_count} 个文件，移除了 {objs_count} 个忽略目标。")
    
    print("\n转换完成！")
    print("注意：生成的标签格式为 '父类:子类 x y w h'，请确保后续训练代码支持此格式。")

if __name__ == '__main__':
    main()
