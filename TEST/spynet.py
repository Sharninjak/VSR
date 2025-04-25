import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence


class SpyNetUnit(nn.Module):

    def __init__(self, input_channels: int = 8):
        super(SpyNetUnit, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3, stride=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 32, kernel_size=7, padding=3, stride=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 16, kernel_size=7, padding=3, stride=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(16, 2, kernel_size=7, padding=3, stride=1))

    def forward(self, 
                frames: tuple[torch.Tensor, torch.Tensor], 
                optical_flow: torch.Tensor = None,
                upsample_optical_flow: bool = True) -> torch.Tensor:
        f_frame, s_frame = frames

        # G的输入是两个图片和对应的光流
        # 在第0层也就是金字塔最上层，输入的光流是[]

        if optical_flow is None:
            # If optical flow is None (k = 0) then create empty one having the
            # same size as the input frames, therefore there is no need to 
            # upsample it later
            upsample_optical_flow = False
            b, c, h, w = f_frame.size()
            optical_flow = torch.zeros(b, 2, h, w, device=s_frame.device)

        # 其他层输入的光流是 上一层光流 的 2倍上采样（size和value都要扩大2倍）
        if upsample_optical_flow:
            optical_flow = F.interpolate(
                optical_flow, scale_factor=2, align_corners=True, 
                mode='bilinear') * 2

        s_frame = spynet.nn.warp(s_frame, optical_flow, s_frame.device)
        s_frame = torch.cat([s_frame, optical_flow], dim=1)
        
        inp = torch.cat([f_frame, s_frame], dim=1)
        # inp 是  f_frame,s_frame_warp,optical_flow
        return self.module(inp)


def train(**kwargs):
    torch.manual_seed(0)
    previous = []
    for k in range(kwargs.pop('levels')):
        previous.append(train_one_level(k, previous, **kwargs))
    # previous 开始为空，最后是一个包含k层的网络

    # 训练完成后保存下来 
    final = spynet.SpyNet(previous)
    torch.save(final.state_dict(), 
               str(Path(kwargs['checkpoint_dir']) / f'final.pt'))


def train_one_level(k: int, 
                    previous: Sequence[spynet.SpyNetUnit],
                    **kwargs) -> spynet.SpyNetUnit:

    print(f'Training level {k}...')

    train_ds, valid_ds = load_data(kwargs['root'], k)
    train_dl, valid_dl = build_dl(train_ds, valid_ds, 
                                  kwargs['batch_size'],
                                  kwargs['dl_num_workers'])

    # 返回当前的网络 和 之前的网络， 比如3层的网络和2层的网络
    current_level, trained_pyramid = build_spynets(
        k, kwargs['finetune_name'], previous)
    
    optimizer = torch.optim.AdamW(current_level.parameters(),
                                  lr=1e-5,
                                  weight_decay=4e-5)
    loss_fn = spynet.nn.EPELoss()

    for epoch in range(kwargs['epochs']):
        train_one_epoch(train_dl, 
                        optimizer,
                        loss_fn,
                        current_level,
                        trained_pyramid,
                        print_freq=999999,
                        header=f'Epoch [{epoch}] [Level {k}]')

    torch.save(current_level.state_dict(), 
               str(Path(kwargs['checkpoint_dir']) / f'{k}.pt'))
    
    return current_level


def train_one_epoch(dl: DataLoader,
                    optimizer: torch.optim.AdamW,
                    criterion_fn: torch.nn.Module,
                    Gk: torch.nn.Module, 
                    prev_pyramid: torch.nn.Module = None, 
                    print_freq: int = 100,
                    header: str = ''):
    Gk.train()
    running_loss = 0.

    if prev_pyramid is not None:
        prev_pyramid.eval()

    for i, (x, y) in enumerate(dl):
        x = x[0].to(device), x[1].to(device)
        y = y.to(device)

        if prev_pyramid is not None:
            with torch.no_grad():
                Vk_1 = prev_pyramid(x)
                Vk_1 = F.interpolate(
                    Vk_1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            Vk_1 = None

        predictions = Gk(x, Vk_1, upsample_optical_flow=False)

        if Vk_1 is not None:
            y = y - Vk_1

        loss = criterion_fn(y, predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % print_freq == 0:
            loss_mean = running_loss / i
            print(f'{header} [{i}/{len(dl)}] loss {loss_mean:.4f}')

    loss_mean = running_loss / len(dl)
    print(f'{header} loss {loss_mean:.4f}')


# 返回当前的网络 和 之前的网络
def build_spynets(k: int, name: str, 
                  previous: Sequence[torch.nn.Module]) \
                      -> Tuple[spynet.SpyNetUnit, spynet.SpyNet]:

    if name != 'none':
        pretrained = spynet.SpyNet.from_pretrained(name, map_location=device)
        current_train = pretrained.units[k]
    else:
        current_train = spynet.SpyNetUnit()
        
    current_train.to(device)
    current_train.train()
    
    if k == 0:
        Gk = None
    else:
        Gk = spynet.SpyNet(previous)
        Gk.to(device)
        Gk.eval()

    return current_train, Gk


def warp(image: torch.Tensor, 
         optical_flow: torch.Tensor,
         device: torch.device = torch.device('cpu')) -> torch.Tensor:

    b, c, im_h, im_w = image.size() 
    
    hor = torch.linspace(-1.0, 1.0, im_w).view(1, 1, 1, im_w)
    hor = hor.expand(b, -1, im_h, -1)

    vert = torch.linspace(-1.0, 1.0, im_h).view(1, 1, im_h, 1)
    vert = vert.expand(b, -1, -1, im_w)

    grid = torch.cat([hor, vert], 1).to(device)

    # optical_flow是对应图像size的，因此首先将其缩放到[-1,1]
    # 再与grid相加
    optical_flow = torch.cat([
        optical_flow[:, 0:1, :, :] / ((im_w - 1.0) / 2.0), 
        optical_flow[:, 1:2, :, :] / ((im_h - 1.0) / 2.0)], dim=1)

    # Channels last (which corresponds to optical flow vectors coordinates)
    grid = (grid + optical_flow).permute(0, 2, 3, 1)
    return F.grid_sample(image, grid=grid, padding_mode='border', 
                         align_corners=True)


# 欧式距离
class EPELoss(torch.nn.Module): #end-point-error (EPE)

    def __init__(self):
        super(EPELoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = (target - pred).pow(2).sum().sqrt()
        return dist.mean()


x = Variable(torch.randn([1, 3, 64, 64]))
y0 = F.interpolate(x, scale_factor=0.5)
y1 = F.interpolate(x, size=[32, 32])

y2 = F.interpolate(x, size=[128, 128], mode="bilinear")

print(y0.shape)
print(y1.shape)
print(y2.shape)

# return:
# torch.Size([1, 3, 32, 32])
# torch.Size([1, 3, 32, 32])
# torch.Size([1, 3, 128, 128])

