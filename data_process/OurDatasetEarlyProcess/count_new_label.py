import json

def count_annotation_changes(log_file_path):
    """
    统计标注修改日志中的增删改数量
    
    Args:
        log_file_path: JSON日志文件路径
    
    Returns:
        dict: 包含统计结果的字典
    """
    # 读取JSON文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    # 初始化统计变量
    total_add = 0
    total_delete = 0
    total_modify = 0
    
    # 遍历数据进行统计
    for item in log_data:
        for image_name, changes in item.items():
            total_add += changes.get('add', 0)
            total_delete += changes.get('delete', 0)
            total_modify += changes.get('modify', 0)
    
    # 返回统计结果
    return {
        'total_add': total_add,
        'total_delete': total_delete,
        'total_modify': total_modify,
        'total_operations': total_add + total_delete + total_modify
    }

# 使用示例
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    log_file_path = r"D:\DeepLearning\Challenger\data\edit_history3.json"
    
    # 统计并显示结果
    stats = count_annotation_changes(log_file_path)
    
    print("=" * 30)
    print("标注修改统计结果")
    print("=" * 30)
    print(f"新增标注数量: {stats['total_add']}")
    print(f"删除标注数量: {stats['total_delete']}")
    print(f"修改标注数量: {stats['total_modify']}")
    print(f"总计操作数量: {stats['total_operations']}")
    print("=" * 30)
