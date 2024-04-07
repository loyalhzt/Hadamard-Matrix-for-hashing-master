import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image
from torchvision.transforms import transforms
from werkzeug.utils import secure_filename
from torch.autograd import Variable
import pre_process as prep

import os

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return '没有文件部分'
        file = request.files['file']
        if file.filename == '':
            return '没有选择文件'
        if file:
            # 保存上传的图片
            # secure_filename 函数会删除文件名中的任何不安全字符，只保留字母、数字、下划线和点。这样可以确保文件名不会对服务器上的文件系统造成安全威胁。
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))

            # 调用后端模型获取相似图片的地址列表
            similar_images = get_similar_images(os.path.join('uploads', filename))

            print(similar_images)

            # 渲染模板并传递相似图片列表
            if len(similar_images) == 0:
                return '没有找到相似图片'
            else:
                return render_template('result.html', similar_images=similar_images)

    return render_template('index.html')

def get_image_path_from_index(top_k_indices):
    # 定义存储 all_image_paths 的文件路径
    all_image_paths_file_path = '../data/miniimagenet/all_image_paths.txt'
    # 打开文件并读取每一行
    with open(all_image_paths_file_path, 'r') as f:
        lines = f.readlines()
    # 创建一个列表，存储所有图像的路径
    all_image_paths = []
    # 遍历数据库文件中的每一行，提取图像路径
    for index, line in enumerate(lines):
        # 提取文件路径
        path = line.split(' ')[0]
        # 找到第二个斜杠的位置
        second_slash_index = path.find('/', path.find('/') + 1)
        # 如果存在第二个斜杠，提取除第二个斜杠之前的内容
        if second_slash_index != -1:
            path = path[second_slash_index:]
        # 添加到列表中
        all_image_paths.append(path.strip())
    # 将列表转换为 NumPy 数组
    all_image_paths_array = np.array(all_image_paths)
    # 获取这五个索引对应的图片地址
    top_k_paths = all_image_paths_array[top_k_indices]
    return top_k_paths


def top_k_similar_images(database_hash, test_hash):
    # binary the hash code
    # R = args.R
    T = 0.1
    database_hash[database_hash < T] = -1
    database_hash[database_hash >= T] = 1
    test_hash[test_hash < T] = -1
    test_hash[test_hash >= T] = 1
    sim = np.dot(database_hash, test_hash.T)  # 使用点积计算数据库哈希码和测试集哈希码之间的相似度。
    ids = np.argsort(-sim, axis=0)  # 对相似度进行排序，获取索引
    # 获取与测试图片最相似的前10张图片的索引
    top_k_indices = ids[:10, 0]
    # 打印相似度最高的前10张图片的相似度
    for i, index in enumerate(top_k_indices):
        print(f"相似度 {i + 1}: {sim[index, 0]}")
    print(top_k_indices)
    # 输出这10张图片的路径
    top_k_paths = get_image_path_from_index(top_k_indices)
    return top_k_paths


def predict_hash_code(model, input):
    model.eval()
    # 使用 transforms.ToTensor() 将图像转换为 PyTorch 张量
    img = Image.open(input)
    # input_tensor = transforms.ToTensor()(Image.open(input))
    transform = prep.image_test(resize_size=255, crop_size=224)

    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0)  # 添加一个批大小维度
    input_tensor = Variable(input_tensor).cuda()

    y = model(input_tensor)
    output = y.data.cpu().float()
    return output.cpu().numpy()


def get_similar_images(upload_image_path):
    input = upload_image_path
    model = torch.load('../model/test.pkl')
    print('Waiting for generate the hash code from test set')
    test_hash = predict_hash_code(model, input)
    database_hash = np.load("../data/imagenet/imagenet_64_database_hash.npy")
    return top_k_similar_images(database_hash, test_hash)
    # 这里是调用后端模型的代码


if __name__ == '__main__':
    app.run(debug=True, host="10.252.119.133")
