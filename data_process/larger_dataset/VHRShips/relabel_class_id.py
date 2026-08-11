import os

# ================= 配置区域 =================

# HBB 标签文件所在路径
hbb_folder = r'D:\DeepLearning\Challenger\data\LargerDataset\VHRShips\hbblabels'

# 类别映射字典
# 注意：键是你上一步生成的类别名称（字符串），值是你想要映射的新层级ID（字符串）
CLASS_MAP = {
    "tug": "0:3",
    "bulkCarrier": "0:3",
    "container": "0:3",
    "tanker": "0:3",
    "undefined": "0:-1",
    "roro": "0:3",
    "dredging": "0:3",
    "dredgerReclamation": "0:3",
    "coaster": "0:3",
    "smallBoat": "0:3",
    "coastGuard": "0:3",
    "oreCarrier": "0:3",
    "generalCargo": "0:3",
    "yatch": "0:3",
    "patrolForce": "0:3",
    "serviceCraft": "0:3",
    "bargePontoon": "0:3",
    "drill": "0:3",
    "other": "0:3",
    "oilTanker": "0:3",
    "floatingDock": "0:3",
    "offshore": "0:3",
    "passanger": "0:3",
    "smallPassanger": "0:3",
    "ferry": "0:3",
    "lpg": "0:3",
    "cruiser": "0:3",
    "destroyer": "0:2",
    "auxilary": "0:3",
    "submarine": "0:3",
    "landing": "0:1",
    "fishing": "0:3",
    "frigate": "0:2",
    "aircraft": "0:0",
}

IGNORE_CLASSES = {

}

# ===========================================

def check_unmapped_classes(folder_path, class_map, ignore_set):
    """
    扫描文件夹中的所有标签文件，找出既未映射也未忽略的类别
    """
    unmapped_keys = set()
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not files:
        print(f"警告: 在路径 {folder_path} 下没有找到 .txt 文件")
        return unmapped_keys
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        # 第一列是类别名称
                        original_key = parts[0]
                        if original_key not in class_map and original_key not in ignore_set:
                            unmapped_keys.add(original_key)
        except Exception as e:
            print(f"读取文件 {filename} 出错: {e}")
            
    return unmapped_keys

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
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                original_key = parts[0]
                
                # 逻辑判断
                if original_key in ignore_set:
                    # 如果在忽略列表中，跳过该行
                    file_ignored += 1
                    needs_update = True 
                    continue
                
                if original_key in class_map:
                    # 直接替换为映射后的字符串（保留冒号）
                    # 例如：将 'tug' 替换为 "0:1"
                    parts[0] = class_map[original_key]
                    needs_update = True
                else:
                    # 如果代码运行到这里，说明 check_unmapped_classes 没拦截住异常数据
                    # 为了防止数据丢失，保持原样并打印警告
                    print(f"警告: 文件 {filename} 中发现未定义类别 '{original_key}'，保持原样。")
                
                new_lines.append(' '.join(parts))
            
            # 只有当内容发生变化时才重写文件
            if needs_update:
                with open(filepath, 'w', encoding='utf-8') as f:
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
        print("\n[错误] 发现未映射且未声明忽略的类别！请补充 CLASS_MAP 配置：")
        for key in sorted(list(unmapped)):
            print(f"    '{key}': '0:0',  # <- 请确认此类别的层级ID")
        print("\n程序已暂停，请修改代码中的 CLASS_MAP 后重新运行。")
        return
    else:
        print("[检查通过] 所有类别 均已在配置中定义。")
    
    # 3. 执行更新
    print(f"\n[2/2] 正在更新标签文件...")
    files_count, objs_count = update_label_file(hbb_folder, CLASS_MAP, IGNORE_CLASSES)
    print(f"  -> 已更新 {files_count} 个文件，移除了 {objs_count} 个忽略目标。")
    
    print("\n转换完成！")
    print("注意：生成的标签格式为 '父类:子类 x y w h' (例如: 0:0 0.5 0.5 0.2 0.2)")

if __name__ == '__main__':
    main()
