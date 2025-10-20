import numpy as np
import torch
import argparse
import os
import torch.nn as nn
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import copy

from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
from sklearn.preprocessing import label_binarize
from torch import optim
from importlib import import_module
from torch.utils.data import DataLoader
from ddrdataset import DDR_dataset
from messdataset import MESS_dataset
from datetime import datetime
import cv2
from functions import progress_bar
from torchnet import meter
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, cohen_kappa_score, precision_score, roc_curve, auc
from sklearn.utils import resample
from thop import profile, clever_format
from efficientnet.multi_model import EfficientNet
from eyepacsdataset import Eyepacs_dataset
from models import vgg_lanet, mobilenetv3_lanet, densenet_lanet, inceptionv3_lanet, resnet_lanet, resnet_ganet, \
    resnet_enhabnet, densenet_habnet, resnet_saliencyhabnet, densenet_saliencyhabnet, vgg_saliencyhabnet
from models.resnet_saliencyhabnet import Bottleneck

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='res50', help='model')
parser.add_argument('--visname', '-vis', default='ddr_on_epepacs ', help='visname')
parser.add_argument('--batch-size', '-bs', default=8, type=int, help='batch-size')
parser.add_argument('--lr', '-lr', default=0.001, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n-cls', default=25, type=int, help='n-classes')
parser.add_argument('--save-dir', '-save-dir', default='./checkpoints', type=str, help='save-dir')
parser.add_argument('--printloss', '-pl', default=20, type=int, help='print-loss')
parser.add_argument('--seed', '-seed', type=int, default=12138)
parser.add_argument('--resume', '-re', type=str, default=None)
parser.add_argument('--test', '-test', type=bool, default=True)
parser.add_argument('--adaloss', '-adaloss', type=bool, default=True)

val_epoch = 1
test_epoch = 5


def parse_args():
    global args
    args = parser.parse_args()

def get_lr(cur, epochs):
    if cur < int(epochs * 0.3):
        lr = args.lr
    elif cur < int(epochs * 0.8):
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    return lr

def get_dynamic_lr(cur, epochs):
    power = 0.9
    lr = args.lr * (1 - cur / epochs) ** power
    return lr


# Grad-CAM 生成函数
# def generate_gradcam(model, input_tensor, target_class, target_layer, task='clf'):
#     model.eval()
#     features = []
#     gradients = []
#
#     def forward_hook(module, input, output):
#         features.append(output)
#
#     def backward_hook(module, grad_in, grad_out):
#         gradients.append(grad_out[0])
#
#     # 选择目标层中的最后一个卷积层
#     if isinstance(target_layer, nn.Sequential):
#         last_block = target_layer[-1]
#         if isinstance(last_block, nn.Module) and hasattr(last_block, 'conv3'):
#             conv_layer = last_block.conv3
#         else:
#             raise ValueError("目标层结构不符合预期")
#     else:
#         conv_layer = target_layer
#
#     # 注册钩子
#     hook_f = conv_layer.register_forward_hook(forward_hook)
#     hook_b = conv_layer.register_backward_hook(backward_hook)
#
#     # 前向传播
#     output = model(input_tensor)
#     if task == 'clf':
#         target = output[0][0, target_class]  # 分类任务输出
#     elif task == 'grad':
#         target = output[1][0, target_class]  # 分级任务输出
#     else:
#         raise ValueError("task 参数必须为 'clf' 或 'grad'")
#
#     # 反向传播
#     model.zero_grad()
#     target.backward()
#
#     if not gradients:
#         raise ValueError("未捕获到梯度，请检查目标层或模型结构")
#
#     # 计算 Grad-CAM
#     feature = features[0]  # [1, C, H, W]
#     gradient = gradients[0]  # [1, C, H, W]
#     weights = gradient.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
#     cam = (weights * feature).sum(dim=1, keepdim=True)  # [1, 1, H, W]
#     cam = F.relu(cam)  # 只保留正向影响
#     cam = cam.squeeze().cpu().detach()  # [H, W]
#
#     # 归一化到 [0, 1]
#     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#
#     # 移除钩子
#     hook_f.remove()
#     hook_b.remove()
#
#     return cam
#
# # 叠加热图到原始图像
# def overlay_heatmap(cam, image, threshold=0.2, alpha=0.6, colormap=cv2.COLORMAP_HOT):
#     """
#     将热图叠加到原始图像上，仅在关注区域（高强度区域）应用热图颜色，其他区域保持原始图像颜色。
#
#     Args:
#         cam: 热图张量 [H, W]
#         image: 原始图像 (PIL Image)
#         threshold: 强度阈值，低于此阈值的区域保持原始图像颜色
#         alpha: 透明度，控制热图与原始图像的融合比例
#         colormap: OpenCV 颜色映射，默认使用 COLORMAP_HOT
#
#     Returns:
#         overlay: 叠加后的图像 (numpy array, BGR格式)
#     """
#     # 将热图转换为 numpy 数组
#     cam = cam.numpy()
#     # 上采样到原始图像大小
#     cam = cv2.resize(cam, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
#     # 应用高斯滤波平滑热图
#     cam = cv2.GaussianBlur(cam, (25, 25), 0)
#     # 归一化到 [0, 1]
#     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#     # 转换为 uint8 格式
#     cam = np.uint8(255 * cam)
#
#     # 生成彩色热图
#     heatmap = cv2.applyColorMap(cam, colormap)
#
#     # 转换原始图像为 numpy 数组并确保格式为 BGR
#     image = np.array(image)
#     if image.ndim == 2 or image.shape[2] == 1:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#
#     # 创建掩码，仅在热图强度超过阈值时应用热图颜色
#     mask = cam > (threshold * 255)
#     overlay = image.copy()
#     overlay[mask] = cv2.addWeighted(image[mask], 1 - alpha, heatmap[mask], alpha, 0)
#
#     return overlay

# def generate_gradcam(model, input_tensor, target_class, target_layer_name, task='clf'):
#     """
#     为特定任务和目标层生成Grad-CAM热图。
#
#     参数:
#         model: PyTorch模型
#         input_tensor: 输入张量 [1, C, H, W]
#         target_class: 目标类别索引
#         target_layer_name: 目标层的名称（字符串）
#         task: 任务类型 ('clf' 为分类, 'grad' 为分级)
#
#     返回:
#         cam: Grad-CAM热图 (numpy数组)
#     """
#     model.eval()
#     features = []
#     gradients = []
#
#     def forward_hook(module, input, output):
#         features.append(output)
#
#     def backward_hook(module, grad_in, grad_out):
#         gradients.append(grad_out[0])
#
#     # 访问目标层
#     if isinstance(model, nn.DataParallel):
#         target_layer = dict(model.module.named_modules())[target_layer_name]
#     else:
#         target_layer = dict(model.named_modules())[target_layer_name]
#
#     # 注册钩子
#     hook_f = target_layer.register_forward_hook(forward_hook)
#     hook_b = target_layer.register_backward_hook(backward_hook)
#
#     # 前向传播
#     output_clf, output_grad = model(input_tensor)
#     if task == 'clf':
#         target = output_clf[0, target_class]
#     elif task == 'grad':
#         target = output_grad[0, target_class]
#     else:
#         raise ValueError("任务必须是 'clf' 或 'grad'")
#
#     # 反向传播
#     model.zero_grad()
#     target.backward()
#
#     if not gradients or not features:
#         hook_f.remove()
#         hook_b.remove()
#         raise ValueError(f"未捕获到 {target_layer_name} 层的梯度或特征")
#
#     # 计算Grad-CAM
#     feature = features[0]  # [1, C, H, W]
#     gradient = gradients[0]  # [1, C, H, W]
#     weights = torch.mean(gradient, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
#     cam = F.relu((weights * feature).sum(dim=1))  # [1, H, W]
#     cam = F.interpolate(cam.unsqueeze(0), size=input_tensor.shape[2:], mode='bilinear',
#                         align_corners=False)  # [1, 1, H, W]
#     cam = cam.squeeze().cpu().detach().numpy()  # [H, W]
#     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化到 [0, 1]
#     cam = cv2.GaussianBlur(cam, (15, 15), 0)  # 平滑热图
#
#     # 移除钩子
#     hook_f.remove()
#     hook_b.remove()
#
#     return cam
#
#
# def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
#     """
#     将热图叠加到原始图像上，生成蓝色背景，高关注度区域为黄色和红色。
#
#     参数:
#         heatmap: Grad-CAM热图 (numpy数组)
#         image: 原始图像 (PIL Image)
#         alpha: 热图透明度，控制叠加强度 (0到1之间，默认0.5)
#         colormap: OpenCV颜色映射 (默认为COLORMAP_JET)
#
#     返回:
#         overlay: 叠加后的图像 (numpy数组)
#     """
#     # 将热图调整为与图像相同的大小
#     heatmap = cv2.resize(heatmap, (image.width, image.height))
#     # 归一化热图到0-255
#     heatmap = np.uint8(255 * heatmap)
#     # 应用颜色映射（COLORMAP_JET：蓝色（低）->黄色->红色（高））
#     heatmap_colored = cv2.applyColorMap(heatmap, colormap)
#
#     # 将PIL图像转换为numpy数组
#     image_np = np.array(image)
#
#     # 确保图像是RGB格式，去除可能的alpha通道
#     if image_np.shape[2] == 4:
#         image_np = image_np[:, :, :3]
#
#     # 创建蓝色背景
#     blue_background = np.zeros_like(image_np, dtype=np.uint8)
#     blue_background[:, :] = [255, 0, 0]  # BGR格式，蓝色为(255, 0, 0)
#
#     # 叠加蓝色背景和热图
#     overlay = cv2.addWeighted(blue_background, 1 - alpha, heatmap_colored, alpha, 0)
#
#     # 将原始图像与叠加结果融合，确保高关注区域突出
#     final_overlay = cv2.addWeighted(image_np, 0.3, overlay, 0.7, 0)
#
#     return final_overlay


# 新增动态损失权重类
# 最初动态损失函数
class DynamicLossWeights:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.avg_loss_clf = 0.0
        self.avg_loss_grad = 0.0

    def update(self, loss_clf, loss_grad):
        # 更新滑动平均
        self.avg_loss_clf = self.alpha * self.avg_loss_clf + (1 - self.alpha) * loss_clf
        self.avg_loss_grad = self.alpha * self.avg_loss_grad + (1 - self.alpha) * loss_grad

    def compute_weights(self):
        # 计算动态权重
        total = self.avg_loss_clf + self.avg_loss_grad + 1e-6  # 避免除零
        weight_clf = self.avg_loss_grad / total
        weight_grad = self.avg_loss_clf / total
        return weight_clf, weight_grad


best_acc = 0
best_kappa_clf = 0
best_kappa_grad = 0
best_auc_clf = 0
best_auc_grad = 0

best_test_acc = 0
best_test_kappa_clf = 0
best_test_kappa_grad = 0
best_test_auc_clf = 0
best_test_auc_grad = 0

def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstraps=1000, alpha=0.05):
    """
    通过bootstrap方法计算95%置信区间
    """
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower = sorted_scores[int((alpha / 2) * len(sorted_scores))]
    upper = sorted_scores[int((1 - alpha / 2) * len(sorted_scores))]

    return lower, upper



def main():
    global best_acc, save_dir

    parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 加载模型
    if args.model == 'res50':
        net = resnet_saliencyhabnet.resnet50(pretrained=True, adaloss=args.adaloss)
    elif args.model == 'res18':
        net = resnet_enhabnet.resnet18(pretrained=True, adaloss=args.adaloss)
    elif args.model == 'effb3':
        net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5, image_size=(512, 512))
    elif args.model == 'dense121':
        net = densenet_saliencyhabnet.densenet121(pretrained=True, adaloss=args.adaloss)
        net.classifier = nn.Linear(1024, 5)
    elif args.model == 'vgg':
        net = vgg_saliencyhabnet.vgg16_bn(pretrained=True, adaloss=args.adaloss)
        net.classifier[6] = nn.Linear(4096, 5)
    elif args.model == 'mobilev3':
        net = mobilenetv3_lanet.mobilenet_v3_large(pretrained=True, adaloss=args.adaloss)
        net.classifier[3] = nn.Linear(1280, 5)
    elif args.model == 'inceptionv3':
        net = inceptionv3_lanet.inception_v3(pretrained=True, aux_logits=False, adaloss=args.adaloss)
        net.fc = nn.Linear(2048, 5)

    if args.adaloss:
        s1 = net.sigma1
        s2 = net.sigma2
    else:
        s1 = torch.zeros(1)
        s2 = torch.zeros(1)

    print(net)
    # exit()

    net = nn.DataParallel(net)
    net = net.cuda()

    net_for_profile = copy.deepcopy(net)
    if isinstance(net_for_profile, torch.nn.DataParallel):
        net_for_profile = net_for_profile.module  # 解包

    net_for_profile = net_for_profile.cuda()  # 确保在 GPU 上

    dummy_input = torch.randn(1, 3, 512, 512).cuda()
    flops, params = profile(net_for_profile, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print("\n===== Model Complexity =====")
    print(f"Total Parameters: {params}")
    print(f"FLOPs: {flops}")

    # 加载数据集
    # ddr
    # dataset = DDR_dataset(train=True, val=False, test=False, multi=args.n_classes)
    # valset = DDR_dataset(train=False, val=True, test=False, multi=args.n_classes)
    # testset = DDR_dataset(train=False, val=False, test=True, multi=args.n_classes)

    # # eyepacs
    dataset = Eyepacs_dataset(train=True, val=False, test=False, multi=args.n_classes)
    valset = Eyepacs_dataset(train=False, val=True, test=False, multi=args.n_classes)
    testset = Eyepacs_dataset(train=False, val=False, test=True, multi=args.n_classes)

    # # idrid
    # dataset = IDRID_dataset(train=True, val=False, test=False, multi=args.n_classes)
    # valset = IDRID_dataset(train=False, val=True, test=False, multi=args.n_classes)
    # testset = IDRID_dataset(train=False, val=False, test=True, multi=args.n_classes)

    # # mess
    # dataset = MESS_dataset(train=True, val=False, test=False, multi=args.n_classes)
    # valset = MESS_dataset(train=False, val=True, test=False, multi=args.n_classes)
    # testset = MESS_dataset(train=False, val=False, test=True, multi=args.n_classes)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, shuffle=False, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    # optim scheduler & crit
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    criterion_clf = nn.CrossEntropyLoss()
    criterion_clf = criterion_clf.cuda()
    criterion_grad = nn.CrossEntropyLoss()
    criterion_grad = criterion_grad.cuda()

    con_matx_clf = meter.ConfusionMeter(2)
    con_matx_grad = meter.ConfusionMeter(5)

    save_dir = './checkpoints/' + args.visname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_log = open('./logs/' + args.visname + '.txt', 'a')

    if args.test:
        #  测试要记得改visname，不然test又分不清是哪个了
        if args.model == 'res50':
            # weight_dir = 'checkpoints/ddr512_res50_camcat_bs32/95.pkl'
            # weight_dir = 'checkpoints/ddr_res50_saliencyhabnet_adl_process_new/64.pkl'
            weight_dir = 'checkpoints/eyepacs_res50_saliencyhabnet_adl_process/45.pkl'
            # weight_dir = 'checkpoints/mess_res50_saliencyhabnet_adl_process/81 .pkl'
        elif args.model == 'res18':
            weight_dir = 'checkpoints/ddr512_res18_camcat_adl_bs32/171.pkl'
        elif args.model == 'effb3':
            # weight_dir = 'checkpoints/ddr512_effb3_camcat_bs32/76.pkl'
            weight_dir = 'checkpoints/ddr512_effb3_camcat_adl_bs32/172.pkl'
        elif args.model == 'vgg':
            # weight_dir = 'checkpoints/ddr512_vgg16_camcat_bs32/65.pkl'
            # weight_dir = 'checkpoints/ddr512_vgg16_camcat_bs32/73.pkl'
            # weight_dir = 'checkpoints/ddr512_vgg16_camcat_bs32/61.pkl'
            weight_dir = 'checkpoints/eyepacs_vgg_saliencyhabnet_adl_process/15.pkl'
            # weight_dir = 'checkpoints/ddr512_vgg19_camcat_adl_bs32/88.pkl'
        elif args.model == 'dense121':
            weight_dir = 'checkpoints/eyepacs_dense121_saliencyhabnet_adl_process/54.pkl'
            # weight_dir = 'checkpoints/ddr_dense121_saliencyhabnet_oldadl_process/71.pkl'

        epoch = int(weight_dir.split('/')[-1].split('.')[0])
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['net']
        net.load_state_dict(state_dict, strict=True)
        test_log = open('./logs/test.txt', 'a')

        ddr_test(net, testloader, optimizer, epoch, test_log)
        exit()

    # resume from one epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            start_epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model loaded from {}'.format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0

    # 损失函数与优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    criterion_clf = nn.CrossEntropyLoss().cuda()
    criterion_grad = nn.CrossEntropyLoss().cuda()

    con_matx_clf = meter.ConfusionMeter(2)
    con_matx_grad = meter.ConfusionMeter(5)

    # 初始化动态权重类
    dynamic_weights = DynamicLossWeights(alpha=0.9)

    # # 定义余弦退火调度器
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        con_matx_clf.reset()
        con_matx_grad.reset()
        net.train()
        total_loss_clf = .0
        total_loss_grad = .0
        total = .0
        correct_clf = .0
        correct_grad = .0

        lr = get_dynamic_lr(epoch, args.epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, (x, label_clf, label_grad) in enumerate(dataloader):
            x = x.float().cuda()
            label_clf = label_clf.cuda()
            label_grad = label_grad.cuda()

            # 模型前向传播
            y_pred_clf, y_pred_grad = net(x)
            # y_pred_clf, y_pred_grad, saliency_map = net(x)

            # 计算损失
            loss_clf = criterion_clf(y_pred_clf, label_clf)
            loss_grad = criterion_grad(y_pred_grad, label_grad)

            # 计算准确率
            prediction_clf = y_pred_clf.max(1)[1]
            prediction_grad = y_pred_grad.max(1)[1]
            acc_clf = prediction_clf.eq(label_clf).float().mean().item()
            acc_grad = prediction_grad.eq(label_grad).float().mean().item()

            # 更新动态权重
            dynamic_weights.update(loss_clf.item(), loss_grad.item())
            weight_clf, weight_grad = dynamic_weights.compute_weights()

            # 动态权重联合损失
            loss = weight_clf * loss_clf + weight_grad * loss_grad

            # # 更新动态权重
            # dynamic_weights.update(loss_clf.item(), loss_grad.item(), acc_clf, acc_grad)
            # weight_clf, weight_grad = dynamic_weights.compute_weights()

            # # 动态权重联合损失
            # loss = weight_clf * loss_clf + weight_grad * loss_grad

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新统计
            total_loss_clf += loss_clf.item()
            total_loss_grad += loss_grad.item()
            prediction_clf = y_pred_clf.max(1)[1]
            prediction_grad = y_pred_grad.max(1)[1]
            correct_clf += prediction_clf.eq(label_clf).sum().item()
            correct_grad += prediction_grad.eq(label_grad).sum().item()
            total += x.size(0)

            # 打印动态权重和损失
            if i % args.printloss == 0:
                print(f"Epoch {epoch}, Step {i}: Loss Clf={loss_clf.item():.3f}, Loss Grad={loss_grad.item():.3f}, "
                      f"Weight Clf={weight_clf:.3f}, Weight Grad={weight_grad:.3f}")

        # # 更新学习率
        # scheduler.step()


        # 每个epoch后打印统计信息
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss Clf: {total_loss_clf / len(dataloader):.4f}, "
              f"Loss Grad: {total_loss_grad / len(dataloader):.4f}, "
              f"Acc Clf: {correct_clf / total:.4f}, Acc Grad: {correct_grad / total:.4f}")

        # 验证模型
        if (epoch + 1) % val_epoch == 0:
            ddr_val(net, valloader, optimizer, epoch, test_log, s1, s2)

def plot_roc_curve(y_true, y_pred_proba, n_classes, save_path):
    """
    Plot ROC curve for binary and multi-class classification

    Args:
        y_true: true labels
        y_pred_proba: predicted probabilities for each class
        n_classes: number of classes
        save_path: path to save the ROC curve plot
    """
    plt.figure(figsize=(10, 8))

    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])  # Use probability of class 1
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='blue', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:0.2f})')

        roc_auc_dict = {'binary': roc_auc}

    else:
        # Multi-class classification
        # Binarize the labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        colors = ['blue', 'red', 'green', 'yellow', 'purple']

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                     label=f'ROC curve of class {i} (AUC = {roc_auc[i]:0.2f})')

        roc_auc_dict = roc_auc

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

    return roc_auc_dict

@torch.no_grad()
def ddr_val(net, valloader, optimizer, epoch, test_log, s1, s2):
    global best_acc
    global best_kappa_clf
    global best_kappa_grad
    global best_auc_clf
    global best_auc_grad


    net = net.eval()
    total_acc = .0
    total_loss = .0
    correct_clf = .0
    correct_grad = .0
    total = .0
    count = .0
    con_matx_clf = meter.ConfusionMeter(2)
    con_matx_grad = meter.ConfusionMeter(5)

    pred_clf_list = []
    label_clf_list = []
    pred_clf_proba_list = []

    pred_grad_list = []
    label_grad_list = []
    pred_grad_proba_list = []

    for i, (x, label_clf, label_grad) in enumerate(valloader):
        x = x.float().cuda()
        label_clf = label_clf.cuda()
        label_grad = label_grad.cuda()

        # 获取显著性图和注意力特征
        y_pred_clf, y_pred_grad = net(x)

    # # 确保保存路径存在
    # feat3_dir = f'./feat3_maps/{args.visname}/test/epoch_{epoch}/'
    # saliency_dir = f'./saliency_maps/{args.visname}/test/epoch_{epoch}/'
    # attention_dir = f'./attention_maps/{args.visname}/test/epoch_{epoch}/'
    # fused_dir = f'./fused_maps/{args.visname}/test/epoch_{epoch}/'
    # os.makedirs(feat3_dir, exist_ok=True)
    # os.makedirs(saliency_dir, exist_ok=True)
    # os.makedirs(attention_dir, exist_ok=True)
    # os.makedirs(fused_dir, exist_ok=True)
    #
    # for i, (x, label_clf, label_grad) in enumerate(testloader):
    #     x = x.float().cuda()
    #     label_clf = label_clf.cuda()
    #     label_grad = label_grad.cuda()
    #
    #     # 获取显著性图和注意力特征
    #     y_pred_clf, y_pred_grad, feat3, saliency_map, attended_features = net(x)
    #     # 假设已修改为：y_pred_clf, y_pred_grad, saliency_map, attended_features = net(x)
    #
    #     # 逐一处理 batch 中的每张图像
    #     batch_size = x.size(0)
    #     for j in range(batch_size):
    #         # 提取单张图像及其输出
    #         feat3 = feat3[j, 0].cpu().detach().numpy()  # [H, W]
    #         saliency = saliency_map[j, 0].cpu().detach().numpy()  # [H, W]
    #         attended = attended_features[j].mean(dim=0).cpu().detach().numpy()  # [H, W]，通道平均
    #         input_img = x[j].permute(1, 2, 0).cpu().detach().numpy()  # [H, W, 3]
    #
    #         # 归一化
    #         feat3_norm = (feat3 - feat3.min()) / (feat3.max() - feat3.min() + 1e-6)
    #         saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    #         attended_norm = (attended - attended.min()) / (attended.max() - attended.min() + 1e-6)
    #         input_img_norm = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-6)
    #         input_img_255 = (input_img_norm * 255).astype(np.uint8)
    #
    #         # 确保输入图像为 3 通道
    #         if input_img_255.shape[2] == 1:
    #             input_img_255 = cv2.cvtColor(input_img_255, cv2.COLOR_GRAY2BGR)
    #
    #         # 调整尺寸
    #         feat3_resized = cv2.resize(feat3_norm, (input_img_255.shape[1], input_img_255.shape[0]))
    #         saliency_resized = cv2.resize(saliency_norm, (input_img_255.shape[1], input_img_255.shape[0]))
    #         attended_resized = cv2.resize(attended_norm, (input_img_255.shape[1], input_img_255.shape[0]))
    #
    #         # 保存中间图图
    #         plt.imshow(feat3_resized, cmap='jet')
    #         plt.colorbar()
    #         plt.axis('off')
    #         plt.savefig(os.path.join(saliency_dir, f'feat3_batch_{i}_idx_{j}.png'), bbox_inches='tight')
    #         plt.close()
    #
    #         # 保存显著性图
    #         plt.imshow(saliency_resized, cmap='jet')
    #         plt.colorbar()
    #         plt.axis('off')
    #         plt.savefig(os.path.join(saliency_dir, f'saliency_batch_{i}_idx_{j}.png'), bbox_inches='tight')
    #         plt.close()
    #
    #         # 保存注意力图
    #         plt.imshow(attended_resized, cmap='jet')
    #         plt.colorbar()
    #         plt.axis('off')
    #         plt.savefig(os.path.join(attention_dir, f'attention_batch_{i}_idx_{j}.png'), bbox_inches='tight')
    #         plt.close()
    #
    #         # 生成融合图像
    #         saliency_colored = cv2.applyColorMap((saliency_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #         attended_colored = cv2.applyColorMap((attended_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #         fused_saliency = cv2.addWeighted(input_img_255, 0.6, saliency_colored, 0.4, 0)
    #         fused_attention = cv2.addWeighted(input_img_255, 0.6, attended_colored, 0.4, 0)
    #
    #         # 保存融合图像
    #         cv2.imwrite(os.path.join(fused_dir, f'fused_saliency_batch_{i}_idx_{j}.png'), fused_saliency)
    #         cv2.imwrite(os.path.join(fused_dir, f'fused_attention_batch_{i}_idx_{j}.png'), fused_attention)

        # 分类和分级预测
        y_pred_clf_proba = F.softmax(y_pred_clf, dim=1)
        y_pred_grad_proba = F.softmax(y_pred_grad, dim=1)

        con_matx_clf.add(y_pred_clf.detach(), label_clf.detach())
        con_matx_grad.add(y_pred_grad.detach(), label_grad.detach())

        _, predicted_clf = y_pred_clf.max(1)
        _, predicted_grad = y_pred_grad.max(1)

        pred_clf_list.extend(predicted_clf.cpu().detach().tolist())
        label_clf_list.extend(label_clf.cpu().detach().tolist())
        pred_clf_proba_list.extend(y_pred_clf_proba.cpu().detach().numpy())

        pred_grad_list.extend(predicted_grad.cpu().detach().tolist())
        label_grad_list.extend(label_grad.cpu().detach().tolist())
        pred_grad_proba_list.extend(y_pred_grad_proba.cpu().detach().numpy())

        total += x.size(0)
        count += 1
        correct_clf += predicted_clf.eq(label_clf).sum().item()
        correct_grad += predicted_grad.eq(label_grad).sum().item()

        progress_bar(i, len(valloader), ' Acc clf: %.3f|  Acc grad: %.3f'
                     % (100. * correct_clf / total, 100. * correct_grad / total))

        # Get probabilities using softmax
        y_pred_clf_proba = F.softmax(y_pred_clf, dim=1)
        y_pred_grad_proba = F.softmax(y_pred_grad, dim=1)

        con_matx_clf.add(y_pred_clf.detach(), label_clf.detach())
        con_matx_grad.add(y_pred_grad.detach(), label_grad.detach())

        _, predicted_clf = y_pred_clf.max(1)
        _, predicted_grad = y_pred_grad.max(1)

        # Store predictions and probabilities
        pred_clf_list.extend(predicted_clf.cpu().detach().tolist())
        label_clf_list.extend(label_clf.cpu().detach().tolist())
        pred_clf_proba_list.extend(y_pred_clf_proba.cpu().detach().numpy())

        pred_grad_list.extend(predicted_grad.cpu().detach().tolist())
        label_grad_list.extend(label_grad.cpu().detach().tolist())
        pred_grad_proba_list.extend(y_pred_grad_proba.cpu().detach().numpy())

        total += x.size(0)
        count += 1
        correct_clf += predicted_clf.eq(label_clf).sum().item()
        correct_grad += predicted_grad.eq(label_grad).sum().item()

        progress_bar(i, len(valloader), ' Acc clf: %.3f|  Acc grad: %.3f'
                     % (100. * correct_clf / total, 100. * correct_grad / total))

    # Convert lists to numpy arrays
    pred_clf_proba_array = np.array(pred_clf_proba_list)
    pred_grad_proba_array = np.array(pred_grad_proba_list)

    # Calculate metrics
    acc_clf = 100.0 * accuracy_score(np.array(label_clf_list), np.array(pred_clf_list))
    kappa_clf = 100.0 * cohen_kappa_score(np.array(label_clf_list), np.array(pred_clf_list), weights='quadratic')

    acc_grad = 100.0 * accuracy_score(np.array(label_grad_list), np.array(pred_grad_list))
    kappa_grad = 100.0 * cohen_kappa_score(np.array(label_grad_list), np.array(pred_grad_list), weights='quadratic')

    precision = 100.0 * precision_score(np.array(label_grad_list), np.array(pred_grad_list), average='micro')
    f1 = 100.0 * f1_score(np.array(label_grad_list), np.array(pred_grad_list), average='micro')

    # Plot ROC curves and get AUC scores
    save_dir = f'./plots/{args.visname}/val/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert label lists to numpy arrays
    label_clf_array = np.array(label_clf_list)
    label_grad_array = np.array(label_grad_list)

    # Plot ROC for classification task
    roc_auc_clf = plot_roc_curve(
        np.array(label_clf_list),
        pred_clf_proba_array,
        2,  # binary classification
        os.path.join(save_dir, f'roc_clf_epoch_{epoch}.png')
    )

    # Plot ROC for grading task
    roc_auc_grad = plot_roc_curve(
        np.array(label_grad_list),
        pred_grad_proba_array,
        5,  # 5-class classification
        os.path.join(save_dir, f'roc_grad_epoch_{epoch}.png')
    )

    # Calculate mean AUC for each task
    mean_auc_clf = np.mean(list(roc_auc_clf.values()))
    mean_auc_grad = np.mean(list(roc_auc_grad.values()))


    print('val epoch:', epoch, ' val acc clf: ', acc_clf, 'kappa clf: ', kappa_clf,
          'val acc grad: ', acc_grad, 'kappa grad: ', kappa_grad)
    print('Classification AUC scores:', roc_auc_clf, 'Mean AUC:', mean_auc_clf)
    print('Grading AUC scores:', roc_auc_grad, 'Mean AUC:', mean_auc_grad)

    test_log.write('Epoch:%d   Acc_clf:%.2f   kappa_clf:%.2f  Acc_grad:%.2f   kappa_grad:%.2f   \n' % (
        epoch, acc_clf, kappa_clf, acc_grad, kappa_grad))
    test_log.write('Classification AUC scores: ' + str(roc_auc_clf) + ' Mean AUC: %.4f\n' % mean_auc_clf)
    test_log.write('Grading AUC scores: ' + str(roc_auc_grad) + ' Mean AUC: %.4f\n' % mean_auc_grad)
    test_log.flush()

    # 检查是否需要保存模型
    save_model = False

    if kappa_grad > best_kappa_grad:  # 如果 kappa_grad 有提升
        best_kappa_grad = kappa_grad
        save_model = True
        print(f"New best kappa_grad: {best_kappa_grad:.4f}")

    save_dir = './checkpoints/' + args.visname + '/'
    if save_model:  # 如果满足保存条件
        print('Saving model...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'auc_clf': roc_auc_clf,
            'auc_grad': roc_auc_grad
        }
        save_name = os.path.join(save_dir, f"{epoch}.pkl")
        torch.save(state, save_name)
        print(f"Model saved to {save_name}")

    avg_val_loss = total_loss / len(valloader)
    return avg_val_loss


@torch.no_grad()
def ddr_test(net, testloader, optimizer, epoch, test_log):
    net = net.eval()
    total_acc = .0
    total_loss = .0
    correct_clf = .0
    correct_grad = .0
    total = .0
    count = .0
    con_matx_clf = meter.ConfusionMeter(2)
    con_matx_grad = meter.ConfusionMeter(5)

    pred_clf_list = []
    label_clf_list = []
    pred_clf_proba_list = []

    pred_grad_list = []
    label_grad_list = []
    pred_grad_proba_list = []

    # # 确保保存路径存在
    # feat3_dir = f'./feat3_maps/{args.visname}/test/epoch_{epoch}/'
    # saliency_dir = f'./saliency_maps/{args.visname}/test/epoch_{epoch}/'
    # attention_dir = f'./attention_maps/{args.visname}/test/epoch_{epoch}/'
    # fused_dir = f'./fused_maps/{args.visname}/test/epoch_{epoch}/'
    # os.makedirs(feat3_dir, exist_ok=True)
    # os.makedirs(saliency_dir, exist_ok=True)
    # os.makedirs(attention_dir, exist_ok=True)
    # os.makedirs(fused_dir, exist_ok=True)

    for i, (x, label_clf, label_grad) in enumerate(testloader):
        x = x.float().cuda()
        label_clf = label_clf.cuda()
        label_grad = label_grad.cuda()
        y_pred_clf, y_pred_grad = net(x)

        # 启用梯度计算以支持 Grad-CAM
        # with torch.enable_grad():
        #     x.requires_grad_(True)
        #     y_pred_clf, y_pred_grad = net(x)
        #
        #     vis_root = f"./visualizations/test/epoch_{epoch}/batch_{i}/"
        #     os.makedirs(vis_root, exist_ok=True)
        #
        #     for j in range(x.size(0)):
        #         input_tensor = x[j:j + 1]
        #         # 使用 detach() 移除梯度并转换为图像
        #         input_img = Image.fromarray(
        #             np.uint8((x[j].detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5)
        #         )
        #
        #         # 分类任务Grad-CAM
        #         target_class_clf = label_clf[j].item()
        #         target_layer_clf = 'layer3.5.conv3'  # ResNet-50中layer3的最后一个卷积层
        #         cam_clf = generate_gradcam(net, input_tensor, target_class_clf, target_layer_clf, task='clf')
        #         overlay_clf = overlay_heatmap(cam_clf, input_img)
        #         cv2.imwrite(os.path.join(vis_root, f"gradcam_clf_sample_{j}.png"), overlay_clf[:, :, ::-1])
        #
        #         # 分级任务Grad-CAM
        #         target_class_grad = label_grad[j].item()
        #         target_layer_grad = 'layer4.2.conv3'  # ResNet-50中layer4的最后一个卷积层
        #         cam_grad = generate_gradcam(net, input_tensor, target_class_grad, target_layer_grad, task='grad')
        #         overlay_grad = overlay_heatmap(cam_grad, input_img)
        #         cv2.imwrite(os.path.join(vis_root, f"gradcam_grad_sample_{j}.png"), overlay_grad[:, :, ::-1])

        # 逐一处理 batch 中的每张图像
        # batch_size = x.size(0)
        # for j in range(batch_size):
        #     # 提取单张图像及其输出
        #     feat3_single = feat3[j].mean(dim=0).cpu().detach().numpy()  # [H, W]，
        #     saliency = saliency_map[j, 0].cpu().detach().numpy()  # [H, W]
        #     attended = attended_features[j].mean(dim=0).cpu().detach().numpy()  # [H, W]，通道平均
        #     input_img = x[j].permute(1, 2, 0).cpu().detach().numpy()  # [H, W, 3]
        #
        #     # 归一化
        #     feat3_norm = (feat3_single - feat3_single.min()) / (feat3_single.max() - feat3_single.min() + 1e-6)
        #     saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
        #     attended_norm = (attended - attended.min()) / (attended.max() - attended.min() + 1e-6)
        #     input_img_norm = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-6)
        #     input_img_255 = (input_img_norm * 255).astype(np.uint8)
        #
        #     # 确保输入图像为 3 通道
        #     if input_img_255.shape[2] == 1:
        #         input_img_255 = cv2.cvtColor(input_img_255, cv2.COLOR_GRAY2BGR)
        #
        #     # 调整尺寸
        #     feat3_resized = cv2.resize(feat3_norm, (input_img_255.shape[1], input_img_255.shape[0]))
        #     saliency_resized = cv2.resize(saliency_norm, (input_img_255.shape[1], input_img_255.shape[0]))
        #     attended_resized = cv2.resize(attended_norm, (input_img_255.shape[1], input_img_255.shape[0]))
        #
        #     # 保存中间图
        #     plt.imshow(feat3_resized, cmap='jet')
        #     plt.colorbar()
        #     plt.axis('off')
        #     plt.savefig(os.path.join(feat3_dir, f'feat3_batch_{i}_idx_{j}.png'), bbox_inches='tight')
        #     plt.close()
        #
        #     # 保存显著性图
        #     plt.imshow(saliency_resized, cmap='jet')
        #     plt.colorbar()
        #     plt.axis('off')
        #     plt.savefig(os.path.join(saliency_dir, f'saliency_batch_{i}_idx_{j}.png'), bbox_inches='tight')
        #     plt.close()
        #
        #     # 保存注意力图
        #     plt.imshow(attended_resized, cmap='jet')
        #     plt.colorbar()
        #     plt.axis('off')
        #     plt.savefig(os.path.join(attention_dir, f'attention_batch_{i}_idx_{j}.png'), bbox_inches='tight')
        #     plt.close()
        #
        #     # 生成融合图像
        #     saliency_colored = cv2.applyColorMap((saliency_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #     attended_colored = cv2.applyColorMap((attended_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #     fused_saliency = cv2.addWeighted(input_img_255, 0.6, saliency_colored, 0.4, 0)
        #     fused_attention = cv2.addWeighted(input_img_255, 0.6, attended_colored, 0.4, 0)
        #
        #     # 保存融合图像
        #     cv2.imwrite(os.path.join(fused_dir, f'fused_saliency_batch_{i}_idx_{j}.png'), fused_saliency)
        #     cv2.imwrite(os.path.join(fused_dir, f'fused_attention_batch_{i}_idx_{j}.png'), fused_attention)

        # 分类和分级预测
        y_pred_clf_proba = F.softmax(y_pred_clf, dim=1)
        y_pred_grad_proba = F.softmax(y_pred_grad, dim=1)

        con_matx_clf.add(y_pred_clf.detach(), label_clf.detach())
        con_matx_grad.add(y_pred_grad.detach(), label_grad.detach())

        _, predicted_clf = y_pred_clf.max(1)
        _, predicted_grad = y_pred_grad.max(1)

        pred_clf_list.extend(predicted_clf.cpu().detach().tolist())
        label_clf_list.extend(label_clf.cpu().detach().tolist())
        pred_clf_proba_list.extend(y_pred_clf_proba.cpu().detach().numpy())

        pred_grad_list.extend(predicted_grad.cpu().detach().tolist())
        label_grad_list.extend(label_grad.cpu().detach().tolist())
        pred_grad_proba_list.extend(y_pred_grad_proba.cpu().detach().numpy())

        total += x.size(0)
        count += 1
        correct_clf += predicted_clf.eq(label_clf).sum().item()
        correct_grad += predicted_grad.eq(label_grad).sum().item()

        progress_bar(i, len(testloader), ' Acc clf: %.3f|  Acc grad: %.3f'
                     % (100. * correct_clf / total, 100. * correct_grad / total))

        # Get probabilities using softmax
        y_pred_clf_proba = F.softmax(y_pred_clf, dim=1)
        y_pred_grad_proba = F.softmax(y_pred_grad, dim=1)

        con_matx_clf.add(y_pred_clf.detach(), label_clf.detach())
        con_matx_grad.add(y_pred_grad.detach(), label_grad.detach())

        _, predicted_clf = y_pred_clf.max(1)
        _, predicted_grad = y_pred_grad.max(1)

        # Store predictions and probabilities
        pred_clf_list.extend(predicted_clf.cpu().detach().tolist())
        label_clf_list.extend(label_clf.cpu().detach().tolist())
        pred_clf_proba_list.extend(y_pred_clf_proba.cpu().detach().numpy())

        pred_grad_list.extend(predicted_grad.cpu().detach().tolist())
        label_grad_list.extend(label_grad.cpu().detach().tolist())
        pred_grad_proba_list.extend(y_pred_grad_proba.cpu().detach().numpy())

        total += x.size(0)
        count += 1
        correct_clf += predicted_clf.eq(label_clf).sum().item()
        correct_grad += predicted_grad.eq(label_grad).sum().item()

        progress_bar(i, len(testloader), ' Acc clf: %.3f|  Acc grad: %.3f'
                     % (100. * correct_clf / total, 100. * correct_grad / total))

    # Convert lists to numpy arrays
    pred_clf_proba_array = np.array(pred_clf_proba_list)
    pred_grad_proba_array = np.array(pred_grad_proba_list)

    # Calculate metrics
    acc_clf = 100.0 * accuracy_score(np.array(label_clf_list), np.array(pred_clf_list))
    kappa_clf = 100.0 * cohen_kappa_score(np.array(label_clf_list), np.array(pred_clf_list), weights='quadratic')

    acc_grad = 100.0 * accuracy_score(np.array(label_grad_list), np.array(pred_grad_list))
    kappa_grad = 100.0 * cohen_kappa_score(np.array(label_grad_list), np.array(pred_grad_list), weights='quadratic')

    precision = 100.0 * precision_score(np.array(label_grad_list), np.array(pred_grad_list), average='micro')
    f1 = 100.0 * f1_score(np.array(label_grad_list), np.array(pred_grad_list), average='micro')

    # Plot ROC curves and get AUC scores
    save_dir = f'./plots/{args.visname}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert label lists to numpy arrays
    label_clf_array = np.array(label_clf_list)
    label_grad_array = np.array(label_grad_list)

    # Plot ROC for classification task
    roc_auc_clf = plot_roc_curve(
        np.array(label_clf_list),
        pred_clf_proba_array,
        2,  # binary classification
        os.path.join(save_dir, f'roc_clf_epoch_{epoch}.png')
    )

    # Plot ROC for grading task
    roc_auc_grad = plot_roc_curve(
        np.array(label_grad_list),
        pred_grad_proba_array,
        5,  # 5-class classification
        os.path.join(save_dir, f'roc_grad_epoch_{epoch}.png')
    )

    # === Calculate confidence intervals ===
    acc_grad_lower, acc_grad_upper = bootstrap_confidence_interval(label_grad_list, pred_grad_list, accuracy_score)
    kappa_grad_lower, kappa_grad_upper = bootstrap_confidence_interval(label_grad_list, pred_grad_list,
                                                                       cohen_kappa_score)
    auc_grad_lower, auc_grad_upper = bootstrap_confidence_interval(
        label_grad_list, pred_grad_list,
        lambda y, p: roc_auc_score(y, np.eye(5)[p], multi_class='ovo')
    )

    print("\n===== Confidence Intervals (95%) =====")
    print(f"Accuracy (Grading): {acc_grad:.2f}% (95% CI: {acc_grad_lower * 100:.2f}% - {acc_grad_upper * 100:.2f}%)")
    print(f"Kappa (Grading): {kappa_grad:.3f} (95% CI: {kappa_grad_lower:.3f} - {kappa_grad_upper:.3f})")
    print(f"AUC (Grading): {roc_auc_grad:.3f} (95% CI: {auc_grad_lower:.3f} - {auc_grad_upper:.3f})")

    # Print results
    print('test epoch:%d   acc clf:%.2f  kappa clf:%.2f  acc grad:%.2f  kappa grad:%.2f ' % (
        epoch, acc_clf, kappa_clf, acc_grad, kappa_grad))
    print('precision:', precision)
    print('f1:', f1)
    print('Classification AUC scores:', roc_auc_clf)
    print('Grading AUC scores:', roc_auc_grad)
    print(con_matx_grad.value())

    # Log results
    test_log.write('checkpoints: ' + args.visname + '\n')
    test_log.write('Epoch:%d   Acc_clf:%.2f   kappa_clf:%.2f  Acc_grad:%.2f   kappa_grad:%.2f\n' % (
        epoch, acc_clf, kappa_clf, acc_grad, kappa_grad))
    test_log.write('Classification AUC scores: ' + str(roc_auc_clf) + '\n')
    test_log.write('Grading AUC scores: ' + str(roc_auc_grad) + '\n')
    test_log.flush()




if __name__ == "__main__":
    main()
