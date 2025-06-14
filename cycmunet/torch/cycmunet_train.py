import math
import logging
import os
import pathlib
import time

import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cuda
import torch.backends.cudnn
import torchvision.utils
from pytorch_msssim import SSIM

from model import CycMuNet
from model.util import converter, normalizer
import dataset
from cycmunet.model import model_arg
from cycmunet.run import train_arg

# ------------------------------------------
# Configs

model_args = model_arg(nf=64,
                       groups=8,
                       upscale_factor=2,
                       format='yuv420',
                       layers=4,
                       cycle_count=3
                       )

train_args = train_arg(
    size=(128, 128),
    pretrained="/root/cycmunet-new/checkpoints/monitor-ugly_2x_l4_c3_epoch_19.pth",
    # dataset_type="video",
    # dataset_indexes=[
    #     "/root/videos/cctv-scaled/index-train-good.txt",
    #     "/root/videos/cctv-scaled/index-train-ugly.txt",
    #     "/root/videos/cctv-scaled/index-train-smooth.txt",
    #     "/root/videos/cctv-scaled/index-train-sharp.txt",
    # ],
    dataset_type="triplet",
    dataset_indexes=[
        "/root/dataset/vimeo_triplet/tri_trainlist.txt"
    ],
    preview_interval=100,
    seed=0,
    lr=0.001,
    start_epoch=1,
    end_epoch=11,
    sparsity=True,
    batch_size=2,
    autocast=False,
    loss_type='rmse',
    save_path='checkpoints',
    save_prefix='triplet',
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True # PyTorch >=2.0后已移除此选项
torch.set_float32_matmul_precision("high") 
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# --------------------------------------
# Start of code
# 设置预览间隔，基于数据集索引数量是否为1或与100互质来决定用100还是101
preview_interval = 100 \
    if (len(train_args.dataset_indexes) == 1 or math.gcd(100, len(train_args.dataset_indexes)) == 1) \
    else 101
# 包含放大倍数、网络层数等信息的保存前缀
save_prefix = f'{train_args.save_prefix}_{model_args.upscale_factor}x_l{model_args.layers}_c{model_args.cycle_count}'
save_path = pathlib.Path(train_args.save_path)

# 决定输出图像的排列方式(1行或3行)
nrow = 1 if train_args.size[0] * 9 > train_args.size[1] * 16 else 3

# 设置随机种子
torch.manual_seed(train_args.seed)
torch.cuda.manual_seed(train_args.seed)

# 设置日志记录器
formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s]: %(message)s')
# 创建一个流处理器
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
# 创建一个训练过程日志记录器
logger = logging.getLogger('train_progress')
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
# 创建初始化日志记录器
logger_init = logging.getLogger('initialization')
logger_init.addHandler(ch)
logger_init.setLevel(logging.DEBUG)

cvt = converter() # 色彩空间转换器 RGB and YUV420/422/444
norm = normalizer() # 归一化工具

dataset_types = {
    'triplet': dataset.ImageSequenceDataset,
    'video': dataset.VideoFrameDataset
} # 图像序列或视频帧
Dataset = dataset_types[train_args.dataset_type]
# 创建数据集
if len(train_args.dataset_indexes) == 1:
    # 如果只有一个数据集索引，则直接创建该数据集
    ds_train = Dataset(train_args.dataset_indexes[0],
                       train_args.size,
                       model_args.upscale_factor,
                       augment=True,
                       seed=train_args.seed)
else:
    # 如果有多个数据集索引，则创建一个交错数据集
    ds_train = dataset.InterleavedDataset(*[
        Dataset(dataset_index,
                train_args.size,
                model_args.upscale_factor,
                augment=True,
                seed=train_args.seed + i)
        for i, dataset_index in enumerate(train_args.dataset_indexes)])
# 创建数据加载器
ds_train = DataLoader(ds_train,
                      num_workers=1,
                      batch_size=train_args.batch_size,
                      shuffle=Dataset.want_shuffle,  # Video dataset friendly
                      drop_last=True) # 丢弃最后一个不完整的batch

model = CycMuNet(model_args) # 创建模型
model.train() # 设置模型为训练模式
model_updated = False # 模型是否更新的标志
num_params = 0
# 遍历模型的所有参数(张量) model.parameters()
for param in model.parameters():
    # 计算张量中元素的总数量
    num_params += param.numel()
logger_init.info(f"Model has {num_params} parameters.")

# 尝试加载预训练权重
if train_args.pretrained:
    if not os.path.exists(train_args.pretrained):
        logger_init.warning(f"Pretrained weight {train_args.pretrained} not exist.")
    state_dict = torch.load(train_args.pretrained, map_location=lambda storage, loc: storage)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        logger_init.warning(f"Unknown parameters ignored: {load_result.unexpected_keys}")
    if load_result.missing_keys:
        logger_init.warning(f"Missing parameters not initialized: {load_result.missing_keys}")
    logger_init.info("Pretrained weights loaded.")

# 将模型移动到GPU上
model = model.cuda()
# 创建 Adamax 优化器，Adamax 是一种基于无穷范数的 Adam 变体
# - model.parameters(): 需要优化的参数，这里传入模型的所有参数
# - betas=(0.9, 0.999): 用于计算梯度和梯度平方的移动平均系数
#   - beta1=0.9: 一阶矩估计的指数衰减率，控制动量
#   - beta2=0.999: 二阶矩估计的指数衰减率，控制学习率自适应缩放
# - eps=1e-8: 添加到分母以提高数值稳定性的小常数，防止除零错误
optimizer = optim.Adamax(model.parameters(), lr=train_args.lr, betas=(0.9, 0.999), eps=1e-8)
# Or, train only some parts
# optimizer = optim.Adamax(itertools.chain(
#         model.head.parameters(),
#         model.fe.parameters(),
#         model.fr.parameters(),
#         model.tail.parameters()
# ), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)


# 定义一个学习率调度器，使用余弦退火warm restarts策略
# - T_0: 学习率周期长度 - eta_min=: 学习率的最小值,学习率会在初始值和这个最小值之间按余弦曲线变化
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40000, eta_min=1e-7)

# 计算训练参数的数量
num_params_train = 0
for group in optimizer.param_groups:
    for params in group.get('params', []):
        # 累加参数数量
        num_params_train += params.numel()
logger_init.info(f"Model has {num_params} parameters to train.")

if train_args.sparsity:
    # 如果训练参数中设置了稀疏性，则导入ASP模块
    from apex.contrib.sparsity import ASP

    target_layers = []
    # 将模型mu中的所有模块名称添加到target_layers列表中
    target_layers.extend('mu.' + name for name, _ in model.mu.named_modules())
    # 将模型fr中的所有模块名称添加到target_layers列表中
    target_layers.extend('fr.' + name for name, _ in model.fr.named_modules())

    # 初始化模型进行剪枝
    ASP.init_model_for_pruning(model,
                               mask_calculator="m4n2_1d",
                               allowed_layer_names=target_layers,
                               verbosity=2,
                               whitelist=[torch.nn.Linear, torch.nn.Conv2d],
                               allow_recompute_mask=False,
                               allow_permutation=False,
                               )
    # 初始化优化器进行剪枝
    ASP.init_optimizer_for_pruning(optimizer)

    # import torch.fx
    # original_symbolic_trace = torch.fx.symbolic_trace
    # torch.fx.symbolic_trace = functools.partial(original_symbolic_trace, concrete_args={
    #     'batch_mode': '_no_use_sparsity_pseudo',
    #     'stop_at_conf': False,
    #     'all_frames': True
    # })

    # 计算稀疏掩码
    ASP.compute_sparse_masks()
    # torch.fx.symbolic_trace = original_symbolic_trace
    logger.info('Training with sparsity.')


epsilon = (1 / 255) ** 2


# rmse计算两个张量a和b之间的均方根误差
def rmse(a, b):
    return torch.mean(torch.sqrt((a - b) ** 2 + epsilon))


# ssim用于计算两个图像的相似度
ssim_module = SSIM(data_range=1.0, nonnegative_ssim=True).cuda()
def ssim(a, b):
    return 1 - ssim_module(a, b)

# 将输入数据转换为cuda类型并指定数据类型为force_data_dtype
def recursive_cuda(li, force_data_dtype):
    # 判断li是否为列表或元组
    if isinstance(li, (list, tuple)):
        # 如果是，则递归调用recursive_cuda函数，将li中的每个元素转换为cuda类型
        return tuple(recursive_cuda(i, force_data_dtype) for i in li)
    else:
        # 如果不是，则判断force_data_dtype是否为None
        if force_data_dtype is not None:
            # 如果不为None，则将li转换为cuda类型，并指定数据类型为force_data_dtype
            return li.cuda().to(force_data_dtype)
        else:
            # 如果为None，则直接将li转换为cuda类型
            return li.cuda()


def train(epoch):
    epoch_loss = 0
    total_iter = len(ds_train) # 获取训练集的长度 ds_train数据加载器
    loss_coeff = [1, 0.5, 1, 0.5] # 定义损失系数
    # progress: tqdm进度条 total_iter: 迭代次数 desc: 进度条描述
    with tqdm.tqdm(total=total_iter, desc=f"Epoch {epoch}") as progress:
        # 遍历训练集
        for it, data in enumerate(ds_train):
            optimizer.zero_grad() # 梯度清零

            # 定义计算损失函数
            def compute_loss(force_data_dtype=None):
                # 将数据放入GPU中 高频分量(hf0,hf1,hf2)和低频分量(lf0,lf1,lf2)
                (hf0, hf1, hf2), (lf0, lf1, lf2) = recursive_cuda(data, force_data_dtype)
                # hf0,hf1,hf2,lf1转换为rgb lf0,lf2转换为yuv
                if Dataset.pix_type == 'yuv':
                    target = [cvt.yuv2rgb(*inp) for inp in (hf0, hf1, hf2, lf1)]
                else:
                    target = [hf0, hf1, hf2, lf1]

                # 当迭代次数是预览间隔的倍数时，生成预览图 只处理批次中的第一张图用于预览
                if it % preview_interval == 0:
                    if Dataset.pix_type == 'yuv':
                        # 将yuv转换为rgb，并插值放大 为了将图像转换为与hf0、hf1、hf2相同的大小
                        # nearest最近邻插值算法，速度快但质量较低，适合预览
                        # detach()将张量从计算图中分离出来,不需要为预览图像计算梯度,cpu()将其移动到CPU上
                        org = [F.interpolate(cvt.yuv2rgb(y[0:1], uv[0:1]),
                                             scale_factor=(model_args.upscale_factor, model_args.upscale_factor),
                                             mode='nearest').detach().float().cpu()
                               for y, uv in (lf0, lf1, lf2)] # 三帧低分辨率
                    else:
                        org = [F.interpolate(lf[0:1],
                                             scale_factor=(model_args.upscale_factor, model_args.upscale_factor),
                                             mode='nearest').detach().float().cpu()
                               for lf in (lf0, lf1, lf2)]
                        
                # lf0,lf2转换为yuv
                if Dataset.pix_type == 'rgb':
                    lf0, lf2 = cvt.rgb2yuv(lf0), cvt.rgb2yuv(lf2)

                t0 = time.perf_counter() # 计时1
                lf0, lf2 = norm.normalize_yuv_420(*lf0), norm.normalize_yuv_420(*lf2) # 归一化
                outs = model(lf0, lf2, batch_mode='batch') # 将处理后的输入数据传入模型进行前向计算 批处理模式

                t1 = time.perf_counter() # 计时2
                actual = [cvt.yuv2rgb(*norm.denormalize_yuv_420(*out)) for out in outs]
                
                # 计算损失
                if train_args.loss_type == 'rmse':
                    loss = [rmse(a, t) * c for a, t, c in zip(actual, target, loss_coeff)]
                elif train_args.loss_type == 'ssim':
                    loss = [ssim(a, t) * c for a, t, c in zip(actual, target, loss_coeff)]
                else:
                    raise ValueError("Unknown loss type: " + train_args.loss_type)

                assert not any(torch.any(torch.isnan(i)).item() for i in loss)

                t2 = time.perf_counter() # 计时3

                # 定期生成预览
                if it % preview_interval == 0:
                    out = [i[0:1].detach().float().cpu() for i in actual[:3]]
                    ref = [i[0:1].detach().float().cpu() for i in target[:3]]

                    for idx, ts in enumerate(zip(org, out, ref)):
                        torchvision.utils.save_image(torch.concat(ts), f"./result/out{idx}.png",
                                                     value_range=(0, 1), nrow=nrow, padding=0)

                return loss, t1 - t0, t2 - t1
            
            if train_args.autocast:
                # 启用自动混合精度AMP
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss, t_forward, t_loss = compute_loss(torch.float16)
            else:
                loss, t_forward, t_loss = compute_loss()

            total_loss = sum(loss)
            epoch_loss += total_loss.item() # .item():将单元素张量转换为Python标量

            t3 = time.perf_counter()
            total_loss.backward() # 自动微分计算梯度（反向传播）
            optimizer.step() # 根据梯度更新模型参数
            scheduler.step() # 更新学习率调度器
            t_backward = time.perf_counter() - t3 # 反向传播时间

            global model_updated
            model_updated = True # 标记模型已更新
            # 更新进度条
            progress.set_postfix(ordered_dict={
                "loss": f"{total_loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                "f": f"{t_forward:.4f}s",
                "l": f"{t_loss:.4f}s",
                "b": f"{t_backward:.4f}s",
            })
            progress.update()

    logger.info(f"Epoch {epoch} Complete: Avg. Loss: {epoch_loss / total_iter:.4f}")


# 保存模型checkpoint
def save_model(epoch):
    if epoch == -1:
        name = "snapshot"
    else:
        name = f"epoch_{epoch}"
    if not os.path.exists(save_path):
        os.makedirs(save_path) # 创建保存路径
    output_path = save_path / f"{save_prefix}_{name}.pth"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Checkpoint saved to {output_path}")


if __name__ == '__main__':
    try:
        for epoch in range(train_args.start_epoch, train_args.end_epoch):
            # with torch.autograd.detect_anomaly():
            #     train(epoch)
            train(epoch)
            save_model(epoch)
    except KeyboardInterrupt:
        if model_updated:
            save_model(-1)
