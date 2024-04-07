import numpy as np
import torchvision
import torch
from options import parser
from data_list import ImageList
import pre_process as prep
from torch.autograd import Variable


def mean_average_precision(database_hash, test_hash, database_labels, test_labels, args):  # R = 1000
    # binary the hash code
    R = args.R
    T = args.T
    database_hash[database_hash < T] = -1
    database_hash[database_hash >= T] = 1
    test_hash[test_hash < T] = -1
    test_hash[test_hash >= T] = 1

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)  # 使用点积计算数据库哈希码和测试集哈希码之间的相似度。
    ids = np.argsort(-sim, axis=0)  # 对相似度进行降序排序， axis=0表示按列排序。根据相似度对数据库中的样本进行排序，得到每个查询的相似度排序索引。
    # data_dir = 'data/' + args.data_name
    # ids_10 = ids[:10, :]

    # np.save(data_dir + '/ids.npy', ids_10)
    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        label = test_labels[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1  # 将标签中的 0 改为 -1，以便与二值化的哈希码匹配。
        idx = ids[:, i]  # 获取当前查询在数据库中的相似度排序索引。
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0  # 计算前 R 个样本中与查询标签匹配的数量。
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  # 计算精确度，即前 R 个样本中匹配样本的比例。

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        # 计算数据库中所有与当前查询标签匹配的样本数量。
        # 计算召回率，即匹配样本数量与数据库中所有相关样本数量的比例。
        # 将召回率添加到 Recall 列表中。
        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float64(all_num)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx


def predict_hash_code(model, data_loader):  # data_loader is database_loader or test_loader
    model.eval()
    is_start = True   # 判断是否是第一次循环。
    for i, (input, label) in enumerate(data_loader):  # enumerate函数将data_loader转为索引序列（二维数组）
        input = Variable(input).cuda()  # 图片
        label = Variable(label).cuda()  # 图片独热编码（类别）
        y = model(input)

        if is_start:
            all_output = y.data.cpu().float()
            all_label = label.float()
            is_start = False
        else:
            all_output = torch.cat((all_output, y.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.float()), 0)
            # 用于将新的输出结果 y.data.cpu().float() 添加到现有的输出列表 all_output 中。
            # torch.cat 是 PyTorch 库中的一个函数，用于将多个张量沿着指定维度合并成一个张量。

    return all_output.cpu().numpy(), all_label.cpu().numpy()


def test_MAP(model, database_loader, test_loader, args):
    print('Waiting for generate the hash code from database')
    database_hash, database_labels = predict_hash_code(model, database_loader)
    file_dir = 'data/' + args.data_name
    da_ha_name = args.data_name + '_' + str(args.hash_bit) + '_database_hash.npy'
    da_la_name = args.data_name + '_' + str(args.hash_bit) + '_database_label.npy'
    np.save(file_dir + '/' + da_ha_name, database_hash)
    np.save(file_dir + '/' + da_la_name, database_labels)
    print(database_hash.shape)
    print(database_labels.shape)
    print('Waiting for generate the hash code from test set')
    test_hash, test_labels = predict_hash_code(model, test_loader)
    te_ha_name = args.data_name + '_' + str(args.hash_bit) + '_test_hash.npy'
    te_la_name = args.data_name + '_' + str(args.hash_bit) + '_test_label.npy'
    np.save(file_dir + '/' + te_ha_name, test_hash)
    np.save(file_dir + '/' + te_la_name, test_labels)
    print(test_hash.shape)
    print(test_labels.shape)
    print('Calculate MAP.....')
    MAP, R, APx = mean_average_precision(database_hash, test_hash, database_labels, test_labels, args)

    return MAP, R, APx


if __name__ == '__main__':
    args = parser.parse_args()  # 解析命令行参数，返回args这个命名空间
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))   # 将命令行参数转为字典

    database_list = 'data/' + args.data_name + '/database.txt'
    test_list = 'data/' + args.data_name + '/test.txt'
    model_name = args.model_name  # or just put your model name here
    model_dir = 'model/' + model_name
    model = torch.load(model_dir)

    if args.data_name == 'imagenet':
        database_list = 'data/miniimagenet/train.txt'
        test_list = 'data/miniimagenet/test.txt'
        num_class = 100
        model = torch.load('model/test.pkl')
    """
    elif args.data_name == 'coco':
        database_list = 'data/coco/database.txt'
        data_name = 'coco'
        test_list = 'data/coco/test.txt'
        num_class = 80
        model = torch.load('data/coco/64_Resnet152_center.pkl')

    elif args.data_name == 'nus_wide':
        database_list = 'data/nus_wide/database.txt'
        data_name = 'nus_wide'
        test_list = 'data/nus_wide/test.txt'
        num_class = 21
        model = torch.load('data/nus_wide/64_Resnet152_center.pkl')
    """

    # 读取图像并进行裁剪
    database = ImageList(open(database_list).readlines(),
                         transform=prep.image_test(resize_size=255, crop_size=224))
    database_loader = torch.utils.data.DataLoader(database, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers)
    #  shuffle如果设置为 True，在每个 epoch 开始时，数据将被随机打乱，有助于模型更好地学习。在测试或验证时，通常设置为 False，以确保按顺序遍历数据。
    #  num_workers=args.workers: 指定用于数据加载的子进程数。通过使用多个子进程，可以提高数据加载的效率。通常，num_workers 的值设置为计算机可用的 CPU 核心数。
    #  database_loader 是一个 data loader对象。 在模型训练期间，database_loader 将在每个训练迭代中提供一个批次的 图像 和 标签，以便模型进行前向传播和反向传播更新参数。

    test_dataset = ImageList(open(test_list).readlines(), transform=prep.image_test(resize_size=255, crop_size=224))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    print('>>>>>>>>>>>>>>>>>>Testing>>>>>>>>>>>>>>>>>>')
    MAP, R, APx = test_MAP(model, database_loader, test_loader, args)

    np.save('data/Apx.npy', np.array(APx))
    print(len(APx))
    print('MAP: %.4f' % MAP)
    print('Recall:%.4f' % R)
