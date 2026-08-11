"""
train 文件目录
[ID: 1] Ship (1050 boxes)
├── [ID: 2] Warship (961 boxes)
│   ├── [ID: 3] Submarine (666 boxes)
│   ├── [ID: 4] Aircraft carrier (30 boxes)
│   │   ├── [ID: 5] Enterprise (69 boxes)
│   │   ├── [ID: 6] Nimitz (84 boxes)
│   │   └── [ID: 7] Midway (13 boxes)
│   ├── [ID: 9] Destroyer (171 boxes)
│   │   ├── [ID:10] Atago DD (167 boxes)
│   │   ├── [ID:11] Arleigh Burke DD (396 boxes)
│   │   ├── [ID:12] Hatsuyuki DD (88 boxes)
│   │   ├── [ID:13] Hyuga DD (70 boxes)
│   │   └── [ID:14] Asagiri DD (48 boxes)
│   ├── [ID:15] Frigate (140 boxes)
│   │   └── [ID:16] Perry FF (423 boxes)
│   ├── [ID:17] Patrol (102 boxes)
│   ├── [ID:18] Landing (69 boxes)
│   │   ├── [ID:19] YuTing LL (61 boxes)
│   │   ├── [ID:20] YuDeng LL (53 boxes)
│   │   ├── [ID:21] YuDao LL (40 boxes)
│   │   ├── [ID:22] YuZhao LL (31 boxes)
│   │   ├── [ID:23] Austin LL (76 boxes)
│   │   ├── [ID:24] Osumi LL (28 boxes)
│   │   ├── [ID:25] Wasp LL (14 boxes)
│   │   ├── [ID:26] LSD_41 LL (95 boxes)
│   │   └── [ID:27] LHA LL (126 boxes)
│   ├── [ID:28] Commander (88 boxes)
│   └── [ID:29] Auxiliary Ships (60 boxes)
│       ├── [ID:30] Medical ship (22 boxes)
│       ├── [ID:31] Test ship (43 boxes)
│       ├── [ID:32] Training ship (31 boxes)
│       ├── [ID:33] AOE (37 boxes)
│       ├── [ID:34] Masyuu AS (28 boxes)
│       ├── [ID:35] Sanantonio AS (48 boxes)
│       └── [ID:36] EPF (42 boxes)
└── [ID:37] Merchant (150 boxes)
    ├── [ID:38] Container Ship (232 boxes)
    ├── [ID:39] RoRo (107 boxes)
    ├── [ID:40] Cargo (657 boxes)
    ├── [ID:41] Barge (161 boxes)
    ├── [ID:42] Tugboat (197 boxes)
    ├── [ID:43] Ferry (191 boxes)
    ├── [ID:44] Yacht (498 boxes)
    ├── [ID:45] Sailboat (325 boxes)
    ├── [ID:46] Fishing Vessel (318 boxes)
    ├── [ID:47] Oil Tanker (129 boxes)
    ├── [ID:48] Hovercraft (229 boxes)
    └── [ID:49] Motorboat (1186 boxes)

[ID:50] DOCK (744 boxes)
"""

import json
import os
from collections import defaultdict

def analyze_coco_file(json_path):
    """分析单个COCO格式标注文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    categories = data['categories']
    annotations = data.get('annotations', [])
    
    # 统计每个类别的标注数量
    cat_annotation_count = defaultdict(int)
    for ann in annotations:
        cat_id = ann['category_id']
        cat_annotation_count[cat_id] += 1
    
    return categories, cat_annotation_count, len(annotations)

def build_hierarchy(categories):
    """构建类别层级关系"""
    # 按照level组织
    hierarchy = {
        'level_0': {},
        'level_1': {},
        'level_2': {}
    }
    
    for cat in categories:
        l0_id = cat.get('level_0_id', 0)
        l1_id = cat.get('level_1_id', 0)
        l2_id = cat.get('level_2_id', 0)
        
        # Level 0
        if l0_id not in hierarchy['level_0']:
            hierarchy['level_0'][l0_id] = {
                'id': cat['id'] if l0_id == cat.get('level_1_id') else None,
                'name': None,
                'supercategory': cat.get('supercategory', 'none')
            }
        if l0_id == cat.get('level_1_id') and l0_id == cat.get('level_2_id'):
            hierarchy['level_0'][l0_id]['name'] = cat['name']
            hierarchy['level_0'][l0_id]['id'] = cat['id']
        
        # Level 1
        if l1_id not in hierarchy['level_1']:
            hierarchy['level_1'][l1_id] = {
                'id': None,
                'name': None,
                'level_0_id': l0_id,
                'supercategory': cat.get('supercategory', 'none'),
                'children': []
            }
        if l1_id == cat.get('level_2_id') and l1_id != l0_id:
            hierarchy['level_1'][l1_id]['name'] = cat['name']
            hierarchy['level_1'][l1_id]['id'] = cat['id']
        
        # Level 2 (叶子节点)
        if l2_id != l1_id:
            hierarchy['level_1'][l1_id]['children'].append({
                'id': cat['id'],
                'name': cat['name'],
                'level_2_id': l2_id,
                'supercategory': cat.get('supercategory', 'none')
            })
    
    return hierarchy

def print_file_statistics(json_path, categories, cat_annotation_count, total_annotations):
    """打印单个文件的统计信息"""
    print(f"\n文件: {os.path.basename(json_path)}")
    print("="*80)
    
    print(f"\n总类别数: {len(categories)}")
    print(f"总标注框数: {total_annotations}")
    
    print("\n类别列表:")
    for cat in categories:
        count = cat_annotation_count.get(cat['id'], 0)
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        print(f"  {cat['id']:2d}. {cat['name']:20s} "
              f"(L0={cat.get('level_0_id', 0)}, L1={cat.get('level_1_id', 0)}, L2={cat.get('level_2_id', 0)}) "
              f": {count:6d} ({percentage:5.2f}%)")

def print_level_comparison(level_files):
    """对比不同层级的类别"""
    print("\n" + "="*80)
    print("各层级类别对比")
    print("="*80)
    
    for level_name, path in sorted(level_files.items()):
        if not os.path.exists(path):
            print(f"\n{level_name}: 文件不存在")
            continue
            
        categories, cat_annotation_count, total_annotations = analyze_coco_file(path)
        
        print(f"\n{level_name.upper().replace('_', ' ')} ({len(categories)} 类别, {total_annotations} 标注框):")
        print("-" * 80)
        
        for cat in categories:
            count = cat_annotation_count.get(cat['id'], 0)
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"  {cat['id']:2d}. {cat['name']:20s} : {count:6d} ({percentage:5.2f}%)")

def print_hierarchy_tree(level_3_path):
    """打印level_3的层级树形结构"""
    if not os.path.exists(level_3_path):
        print(f"\nLevel 3 文件不存在: {level_3_path}")
        return
    
    categories, cat_annotation_count, total_annotations = analyze_coco_file(level_3_path)
    
    print("\n" + "="*80)
    print("Level 3 层级树形结构")
    print("="*80)
    
    # 构建父子关系
    cat_by_id = {cat['id']: cat for cat in categories}
    children = defaultdict(list)
    roots = []
    
    for cat in categories:
        supercat = cat.get('supercategory', 'none')
        if supercat == 'none':
            roots.append(cat)
        else:
            children[supercat].append(cat)
    
    # 递归打印树
    def print_tree(cat, prefix="", is_last=True):
        count = cat_annotation_count.get(cat['id'], 0)
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}[ID:{cat['id']:2d}] {cat['name']} ({count} boxes)")
        
        cat_children = children.get(cat['name'], [])
        for i, child in enumerate(cat_children):
            is_last_child = (i == len(cat_children) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(child, new_prefix, is_last_child)
    
    for root in roots:
        count = cat_annotation_count.get(root['id'], 0)
        print(f"\n[ID:{root['id']:2d}] {root['name']} ({count} boxes)")
        
        root_children = children.get(root['name'], [])
        for i, child in enumerate(root_children):
            is_last = (i == len(root_children) - 1)
            print_tree(child, "", is_last)

def main():
    # 所有层级的文件路径
    level_files = {
        'level_0': r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\labels\COCO_Format\ShipImageNet_train_bnbox_level_0.json",
        'level_1': r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\labels\COCO_Format\ShipImageNet_train_bnbox_level_0.json",
        'level_2': r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\labels\COCO_Format\ShipImageNet_train_bnbox_level_2.json",
        'level_3': r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\labels\COCO_Format\ShipImageNet_train_bnbox_level_3.json",
    }
    
    # 其他数据集划分
    other_files = [
        r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\labels\COCO_Format\ShipImageNet_train_bnbox_level_3.json",
        r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\labels\COCO_Format\ShipImageNet_val_bnbox_level_3.json",
    ]
    
    # 1. 对比各层级
    print_level_comparison(level_files)
    
    # 2. 打印level_3的树形结构
    print_hierarchy_tree(level_files['level_3'])
    
    # 3. 统计train/val/test的level_3
    print("\n" + "="*80)
    print("Train/Val/Test Level 3 统计")
    print("="*80)
    
    all_files = [level_files['level_3']] # + other_files
    for path in all_files:
        if os.path.exists(path):
            categories, cat_annotation_count, total_annotations = analyze_coco_file(path)
            print_file_statistics(path, categories, cat_annotation_count, total_annotations)

if __name__ == "__main__":
    main()
