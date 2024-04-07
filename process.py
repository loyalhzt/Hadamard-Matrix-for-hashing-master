import numpy as np
import torch
from options import parser
from data_list import ImageList
import pre_process as prep
from torch.autograd import Variable

torch.nn.Module.dump_patches = True


def get_image_path_from_index(top_k_indices):
    # 定义存储 all_image_paths 的文件路径
    all_image_paths_file_path = 'data/miniimagenet/all_image_paths.txt'
    # 打开文件并读取每一行
    with open(all_image_paths_file_path, 'r') as f:
        lines = f.readlines()
    # 创建一个列表，存储所有图像的路径
    all_image_paths = []
    # 遍历数据库文件中的每一行，提取图像路径
    for index, line in enumerate(lines):
        # 提取文件路径
        path = line.split(' ')[0]
        all_image_paths.append(path)
    # 将列表转换为 NumPy 数组
    all_image_paths_array = np.array(all_image_paths)
    # 获取这五个索引对应的图片地址
    top_k_paths = all_image_paths_array[top_k_indices]
    return top_k_paths


def top_k_similar_images(database_hash, test_hash):
    # binary the hash code
    # R = args.R
    T = 0
    database_hash[database_hash < T] = -1
    database_hash[database_hash >= T] = 1
    test_hash[test_hash < T] = -1
    test_hash[test_hash >= T] = 1
    sim = np.dot(database_hash, test_hash.T)  # 使用点积计算数据库哈希码和测试集哈希码之间的相似度。
    ids = np.argsort(-sim, axis=0)  # 对相似度进行排序，获取索引
    # 获取与测试图片最相似的前五张图片的索引
    top_k_indices = ids[:5, 0]
    print(top_k_indices)
    # 输出这五张图片的路径
    top_k_paths = get_image_path_from_index(top_k_indices)
    for path in top_k_paths:
        print(path.strip())
    return top_k_paths


def predict_hash_code(model, data_loader):
    model.eval()

    for i, (input, label) in enumerate(data_loader):
        input = Variable(input).cuda()
        print(input)
        y = model(input)
        output = y.data.cpu().float()
        label = Variable(label).cuda()
    return output.cpu().numpy()


if __name__ == '__main__':
    args = parser.parse_args()  # 解析命令行参数，返回args这个命名空间
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))  # 将命令行参数转为字典

    model = torch.load('model/test.pkl')
    test_list = 'data/miniimagenet/test1.txt'

    test_dataset = ImageList(open(test_list).readlines(), transform=prep.image_test(resize_size=255, crop_size=224))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)
    print('Waiting for generate the hash code from test set')

    test_hash = predict_hash_code(model, test_loader)
    print(test_hash.shape)
    database_hash = np.load("data/imagenet/imagenet_64_database_hash.npy")
    database_labels = np.load("data/imagenet/imagenet_64_database_label.npy")
    top_k_similar_images(database_hash, test_hash)
