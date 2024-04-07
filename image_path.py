# 定义数据库文件路径
database_file_path = 'data/miniimagenet/train.txt'

# 加载数据库文件
database_file = open(database_file_path, 'r')
database_lines = database_file.readlines()
database_file.close()

# 创建一个列表，存储所有图像的路径
all_image_paths = []

# 遍历数据库文件中的每一行，提取图像路径
for index, line in enumerate(database_lines):
    # 提取文件路径
    path = line.split(' ')[0]
    all_image_paths.append(path)

# 将 all_image_paths 列表保存到文件中
with open('all_image_paths.txt', 'w') as f:
    for path in all_image_paths:
        f.write(path + '\n')