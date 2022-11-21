'''
    use: python gradcam --image-path xxxxx.png
    来源: https://blog.csdn.net/bairw_Bella/article/details/110161987

'''
import os
import argparse
import cv2
import numpy as np
import torch
from torch import  nn
from torch.autograd import Function
from torchvision import models


os.environ['CUDA_VISIBLE_DEVICES']='2,3'

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():  ##遍历目标层的每一个模块，比如卷积、BN,ReLU
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)  # 利用hook来记录目标层的梯度
                outputs += [x]
        return outputs, x  # output保留目标层的所有模块的特征输出


'''
根据模型结构以及模型的预训练参数，依次对输入图像进行每一个网络层的处理
'''


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
        # 当到达指定输出的网络层时，输出该层的激活特征值并保留该层的输出

            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x
        '''
        x表示对模型最终分类层的输出
        '''


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    # transpose HWC > CHW
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())


        '''
        得到模型最后一层输出特征的分类概率,根据输出的概率的大小判断它与输入图片中哪些区域的像素值更敏感
        '''
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()  # 梯度清零
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)  # 反向传播取得目标层梯度

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        # 调用函数get_gradients(),  得到目标层求得的梯度（目标层特征图的每个像素的偏导数）
        '''
        '''
        target = features[-1]  # feature 包含卷积、BN、和ReLU层，这里取ReLU作为最后的特征值

        target = target.cpu().data.numpy()[0, :]
        # 忽略Batch数目，只取C,H,W维度
        print(target.shape)

        # 特征图每个像素的偏导数求出来之后，取一次宽高维度上的全局平均，得到c类相对于该卷积层输出特征图的第k个通道的敏感程度 weifhts.shape=[n,c]
        weights = np.mean(grads_val, axis=(2, 3))[0, :]

        print(weights.shape)
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        # target[i, :, :] = array:shape(H, W)累加各个通道的特征值
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        print(cam.shape)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str,
                        default='/home3/HWGroup/zhengyx/JY_file/1 MSCMR orient/code/data_transform/C0/5/patient1_C00.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    '''
    加载模型及模型的预训练参数
    '''
    # model = models.resnet50(pretrained=True)
    # 确定模型结构
    model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 16 * 16, 64), nn.Sigmoid(),
    nn.Linear(64, 8)
)
    # 加载参数
    PATH ='/home3/HWGroup/zhengyx/JY_file/1 MSCMR orient/code/checkpoints/C0/model-best.pth'
    checkpoint = torch.load(PATH, map_location={'cuda:2': 'cuda:0','cuda:1': 'cuda:0'})
    # 应用到网络结构中
    model.load_state_dict(checkpoint)
    '''
    指定可视化的卷积层，及卷积层中哪个模块的输出
    '''

    for name, child in model._modules.items():
        print(name)
    grad_cam = GradCam(model=model, feature_module=model[6], target_layer_names=["2"], use_cuda=args.use_cuda)

    '''
    读取图片归一化为[0,1]矩阵,并转换为张量形式
    '''

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (256, 256))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)  # 输出该张图像在指定网络层上，所有卷积通道的特征图叠加后的mask，用于后续和图片的叠加显示

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    print(model._modules.items())
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)