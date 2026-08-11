# 重新映射
# 标注中11-600多全是空图像没有实例 这是正常现象
import os

# ================= 配置区域 =================

# HBB 和 OBB 标签文件所在路径
hbb_folder = r'D:\DeepLearning\Challenger\data\LargerDataset\HRSC2016\hbblabels'
obb_folder = r'D:\DeepLearning\Challenger\data\LargerDataset\HRSC2016\obblabels'

# 类别映射字典
# 格式: '原始Class_ID': '新Class_ID'
# 注意：所有目标的大类均为0，冒号后面是小类
# 请在此处填写完整的映射关系
CLASS_MAP = {
    '100000001': '0:-1',
    '100000002': '0:0',
    '100000003': '0:-1',
    '100000004': '0:3',
    '100000005': '0:0',
    '100000006': '0:0',
    '100000012': '0:0',
    '100000013': '0:0',
    '100000032': '0:0',
    '100000007': '0:2',
    '100000008': '0:1',
    '100000009': '0:2',
    '100000010': '0:1',
    '100000011': '0:3',
    '100000015': '0:1',
    '100000016': '0:1',
    '100000017': '0:3',
    '100000019': '0:3',
    '100000028': '0:3',
    '100000018': '0:3',
    '100000020': '0:3',
    '100000022': '0:3',
    '100000024': '0:3',
    '100000025': '0:3',
    '100000026': '0:3',
    '100000029': '0:3',
    '100000030': '0:3',
}

# 2. 需要忽略的类别
IGNORE_CLASSES = {
    '100000027',
}

# ===========================================

def check_unmapped_classes(folder_path, class_map, ignore_set):
    """
    扫描文件夹中的所有标签文件，找出既未映射也未忽略的类别ID
    """
    unmapped_ids = set()
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        original_id = parts[0]
                        # 只有当ID不在映射表且不在忽略列表时，才视为错误
                        if original_id not in class_map and original_id not in ignore_set:
                            unmapped_ids.add(original_id)
        except Exception as e:
            print(f"读取文件 {filename} 出错: {e}")
            
    return unmapped_ids

def update_label_file(folder_path, class_map, ignore_set):
    """
    更新指定文件夹下的所有标签文件
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
                    # 如果在忽略列表中，跳过该行（不添加到new_lines）
                    file_ignored += 1
                    needs_update = True # 文件内容变化了，需要重写
                    continue
                
                if original_id in class_map:
                    # 如果在映射表中，替换ID
                    parts[0] = class_map[original_id]
                    needs_update = True
                else:
                    # 既不在映射也不在忽略，保留原样并报警
                    print(f"警告: 文件 {filename} 中发现未处理的ID: {original_id}，已保留原ID。")
                
                new_lines.append(' '.join(parts))
            
            # 只有当内容发生变化时才重写文件
            if needs_update:
                # 如果过滤后没有标签了，写入空文件
                with open(filepath, 'w') as f:
                    f.write('\n'.join(new_lines))
                updated_count += 1
                ignored_count += file_ignored
                
        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")
            
    return updated_count, ignored_count

def main():
    print("=" * 50)
    print("开始执行类别映射检查与转换程序")
    print("=" * 50)
    
    # 1. 检查 HBB 文件夹
    print(f"\n[1/4] 正在检查 HBB 路径: {hbb_folder}")
    hbb_unmapped = check_unmapped_classes(hbb_folder, CLASS_MAP, IGNORE_CLASSES)
    
    # 2. 检查 OBB 文件夹
    print(f"[2/4] 正在检查 OBB 路径: {obb_folder}")
    obb_unmapped = check_unmapped_classes(obb_folder, CLASS_MAP, IGNORE_CLASSES)
    
    # 合并未映射的ID
    all_unmapped = hbb_unmapped.union(obb_unmapped)
    
    if all_unmapped:
        print("\n[错误] 发现未映射且未声明忽略的类别 ID！请补充配置：")
        for uid in sorted(list(all_unmapped)):
            print(f"  - 缺少配置: '{uid}'")
            print(f"    -> 如果需要保留，请加入 CLASS_MAP")
            print(f"    -> 如果需要丢弃，请加入 IGNORE_CLASSES")
        print("\n程序已暂停，请修改配置后重新运行。")
        return # 终止程序
    else:
        print("[检查通过] 所有类别 ID 均已在配置中定义。")
    
    # 3. 执行更新
    print("\n[3/4] 正在更新 HBB 标签...")
    hbb_files, hbb_ignored = update_label_file(hbb_folder, CLASS_MAP, IGNORE_CLASSES)
    print(f"  -> 已更新 {hbb_files} 个文件，忽略了 {hbb_ignored} 个目标。")
    
    print("[4/4] 正在更新 OBB 标签...")
    obb_files, obb_ignored = update_label_file(obb_folder, CLASS_MAP, IGNORE_CLASSES)
    print(f"  -> 已更新 {obb_files} 个文件，忽略了 {obb_ignored} 个目标。")
    
    print("\n转换完成！")
    print("提示：请确认 IGNORE_CLASSES 中的类别已正确从标签文件中移除。")

if __name__ == '__main__':
    main()
