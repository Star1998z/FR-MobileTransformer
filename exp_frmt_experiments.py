# -*- coding: utf-8 -*-
"""FR-MobileTransformer 论文实验主脚本
====================================================
基于原始 FRMobileTransformer_Full (2).py 的最小侵入式改造，支持：

  [1] 独立重复实验   --seed 42 / 2026 / 1024 分别运行
  [2] 消融实验       --ablate baseline / reparam / fgfm / clag / aux / full
  [3] Transformer 层数实验 --layers 1 / 2
  [4] 对比实验       --backbone mobilenet_v3_small / mobilenet_v2 /
                     shufflenet_v2_x1_0 / efficientnet_b0 / resnet18
  [5] 验证集划分     --val_ratio 0.15（修复原代码 val=test 的信息泄漏）

与原代码保持一致的关键配置（保证新实验与旧 98.33% 结果可互相引用）：
  - 增强：RandomResizedCrop(0.7-1.0)+HFlip0.5+VFlip0.3+Rot30+ColorJitter+RandAugment(3,10)+Mixup(0.2,0.5)
  - 损失：CE(label_smoothing=0.1) + 0.5*Focal(gamma=2) + 0.3*aux_CE
  - 优化：AdamW(lr=5e-4 全参, wd=0.05, betas=(0.9,0.999))，Warmup5+Cosine 100ep，clip=5.0
  - EMA decay=0.999；早停 patience=15；batch 32(训练,drop_last)/64(验证与测试)
  - 注意：原代码分层学习率因无参数名含 'backbone' 实际未生效（全部 5e-4），此处保持一致

用法（Linux / AutoDL）：
  python exp_frmt_experiments.py --data_path /root/autodl-tmp/project/data_clean --seed 42 --ablate full
  python exp_frmt_experiments.py --data_path ... --seed 2026 --ablate full
  python exp_frmt_experiments.py --data_path ... --seed 42 --ablate baseline
  python exp_frmt_experiments.py --data_path ... --seed 42 --layers 2
  python exp_frmt_experiments.py --data_path ... --seed 42 --backbone mobilenet_v3_small
输出：experiments/<tag>/best_model_full.pth + summary.json + history.json
"""
import os, argparse, json, time, random, math, csv
from datetime import datetime
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# 随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# 模型组件（与原代码一致，增加消融开关 multi_branch / use_fgfm / use_clag / use_aux / transformer_layers）
# ============================================================
class FGFM(nn.Module):
    """频域门控融合模块"""
    def __init__(self, dim):
        super().__init__()
        self.conv_gate = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.freq_mask = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        x_freq = torch.fft.fft2(x, norm='ortho')
        x_freq_masked = x_freq * self.freq_mask
        x_high = torch.fft.ifft2(x_freq_masked, norm='ortho').real
        return x * self.conv_gate(x_high) + x


class ReparamConv(nn.Module):
    """结构重参数化卷积；multi_branch=False 时退化为普通 3x3 卷积（用于 Baseline 消融）"""
    def __init__(self, in_channels, out_channels, stride=1, multi_branch=True):
        super().__init__()
        self.stride = stride
        self.multi_branch = multi_branch
        self.branch_3x3 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        if multi_branch:
            self.branch_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            self.identity = nn.Identity() if in_channels == out_channels and stride == 1 else None

    def forward(self, x):
        out = self.branch_3x3(x)
        if self.multi_branch:
            out = out + self.branch_1x1(x)
            if self.identity:
                out = out + self.identity(x)
        return out


class CrossLevelAttentionGate(nn.Module):
    """跨级注意力门控 (CLAG)"""
    def __init__(self, shallow_dim, deep_dim):
        super().__init__()
        self.W_g = nn.Linear(deep_dim, shallow_dim)
        self.W_x = nn.Conv2d(shallow_dim, shallow_dim, 1)
        self.psi = nn.Sequential(nn.Conv2d(shallow_dim, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.last_attn_map = None

    def forward(self, x_shallow, g_deep):
        g_proj = self.W_g(g_deep).unsqueeze(2).unsqueeze(3)
        self.last_attn_map = self.psi(F.relu(self.W_x(x_shallow) + g_proj))
        return x_shallow * self.last_attn_map


class FRMobileTransformer(nn.Module):
    def __init__(self, num_classes=39, use_fgfm=True, use_clag=True, use_aux=True,
                 multi_branch=True, transformer_layers=1):
        super().__init__()
        self.use_fgfm = use_fgfm
        self.use_clag = use_clag
        self.use_aux = use_aux
        self.stem = nn.Sequential(
            ReparamConv(3, 24, stride=2, multi_branch=multi_branch), nn.BatchNorm2d(24), nn.ReLU(),
            ReparamConv(24, 48, stride=2, multi_branch=multi_branch), nn.BatchNorm2d(48), nn.ReLU())
        self.fgfm = FGFM(48)
        self.downsample = nn.Sequential(
            ReparamConv(48, 96, stride=2, multi_branch=multi_branch), nn.BatchNorm2d(96), nn.ReLU(),
            ReparamConv(96, 160, stride=2, multi_branch=multi_branch), nn.BatchNorm2d(160), nn.ReLU())
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=160, nhead=4, dim_feedforward=320,
                                       batch_first=True, dropout=0.2),
            num_layers=transformer_layers)
        self.clag = CrossLevelAttentionGate(shallow_dim=48, deep_dim=160)
        in_dim = 208 if use_clag else 160
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_dim, num_classes))
        self.aux_classifier = nn.Linear(160, num_classes)

    def forward(self, x, return_features=False):
        x_s = self.stem(x)                      # [B,48,56,56]
        x_s_freq = self.fgfm(x_s) if self.use_fgfm else x_s
        x_d = self.downsample(x_s_freq)         # [B,160,14,14]
        x_seq = x_d.flatten(2).transpose(1, 2)  # [B,196,160]
        x_trans = self.transformer(x_seq)
        g_deep = x_trans.mean(dim=1)            # [B,160]
        if self.use_clag:
            x_gated = self.clag(x_s_freq, g_deep)
            g_shallow = x_gated.mean(dim=[2, 3])
            combined = torch.cat([g_deep, g_shallow], dim=1)  # [B,208]
        else:
            combined = g_deep
        output = self.classifier(combined)
        aux_output = self.aux_classifier(g_deep) if self.use_aux else None
        if return_features:
            return output, aux_output, combined
        return output, aux_output


# ============================================================
# 对比模型（torchvision 预训练，替换分类头；GhostNet/MobileViT 需 timm 时另加）
# ============================================================
def _make_backbone(factory_fn, weights_factory=None, **kwargs):
    """兼容 torchvision 新旧版本 + 权重下载失败时回退随机初始化。

    优先级：
      1) torchvision>=0.13：weights=ImageNet1K_V1（下载失败则 weights=None 随机初始化）
      2) torchvision<0.13：pretrained=True（失败则 pretrained=False）
    """
    try:
        if weights_factory is not None:
            try:
                w = weights_factory.IMAGENET1K_V1
            except Exception:
                w = None
            return factory_fn(weights=w, **kwargs)
        return factory_fn(**kwargs)
    except TypeError:
        # torchvision < 0.13，不支持 weights= 参数
        try:
            return factory_fn(pretrained=True, **kwargs)
        except Exception:
            return factory_fn(pretrained=False, **kwargs)
    except Exception:
        # 权重下载失败/网络超时等：随机初始化兜底，保证实验可继续
        try:
            if weights_factory is not None:
                return factory_fn(weights=None, **kwargs)
        except TypeError:
            return factory_fn(pretrained=False, **kwargs)
        return factory_fn(**kwargs)


def build_backbone(name, num_classes):
    if name == 'mobilenet_v3_small':
        m = _make_backbone(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == 'mobilenet_v2':
        m = _make_backbone(models.mobilenet_v2, models.MobileNet_V2_Weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == 'shufflenet_v2_x1_0':
        m = _make_backbone(models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == 'efficientnet_b0':
        m = _make_backbone(models.efficientnet_b0, models.EfficientNet_B0_Weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == 'resnet18':
        m = _make_backbone(models.resnet18, models.ResNet18_Weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f'unknown backbone: {name}')
    return m


def forward_model(model, images):
    """统一模型调用：FRMT 返回 (out, aux)；torchvision backbone 只返回 logits → (out, None)。"""
    out = model(images)
    if isinstance(out, (tuple, list)):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


# ============================================================
# 训练组件（与原代码一致）
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, aux_weight=0.3, label_smoothing=0.1, focal_weight=0.5):
        super().__init__()
        self.aux_weight = aux_weight
        self.focal_weight = focal_weight
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.focal_loss = FocalLoss(gamma=2.0)

    def forward(self, pred, aux_pred, target):
        ce_loss = self.ce_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        main_loss = ce_loss + self.focal_weight * focal_loss
        aux_loss = 0.0 if aux_pred is None else self.ce_loss(aux_pred, target)
        total = main_loss + self.aux_weight * aux_loss
        return total, main_loss, aux_loss


class WarmupLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, final_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# ============================================================
# 数据与增强（与原代码完全一致）
# ============================================================
def build_transforms():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean, std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandAugment(num_ops=3, magnitude=10),
        transforms.ToTensor(), norm])
    test_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), norm])
    return train_tf, test_tf


def save_curves(history, path):
    """把已跑完的 epoch 画成曲线图（每 5 epoch 自动更新，可中途下载查看）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ep = list(range(1, len(history['train_acc']) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].plot(ep, [a * 100 for a in history['train_acc']], label='Train Acc')
    axes[0].plot(ep, [a * 100 for a in history['val_acc']], label='Val Acc')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history['train_loss'], label='Train Loss')
    axes[1].plot(ep, history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle('FR-MobileTransformer Training Curves (live)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# 训练主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='FR-MobileTransformer 论文实验（重复/消融/层数/对比）')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ablate', type=str, default='full',
                        choices=['full', 'aux', 'clag', 'fgfm', 'reparam', 'baseline'],
                        help='消融配置：full=全模块；aux=去辅助损失；clag=再去CLAG；'
                             'fgfm=再去FGFM；reparam=再去重参数化(单分支)；baseline=仅conv+Transformer主干')
    parser.add_argument('--layers', type=int, default=1, help='Transformer 层数（层数实验 1 vs 2）')
    parser.add_argument('--backbone', type=str, default='frmt',
                        choices=['frmt', 'mobilenet_v3_small', 'mobilenet_v2',
                                 'shufflenet_v2_x1_0', 'efficientnet_b0', 'resnet18'],
                        help='对比实验选择模型；frmt 为 FR-MobileTransformer')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='从训练集划出验证集比例（修复 val=test）')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device} | seed: {args.seed} | ablate: {args.ablate} | layers: {args.layers} | backbone: {args.backbone}')

    # 实验标签
    if args.backbone != 'frmt':
        tag = f'backbone_{args.backbone}_s{args.seed}'
    elif args.layers != 1:
        tag = f'layers{args.layers}_s{args.seed}'
    else:
        tag = f'{args.ablate}_s{args.seed}'
    out_dir = os.path.join('experiments', tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f'输出目录: {out_dir}')

    # ---- 数据 ----
    train_tf, test_tf = build_transforms()
    train_root = os.path.join(args.data_path, 'train')
    test_root = os.path.join(args.data_path, 'test')
    train_full = datasets.ImageFolder(train_root, train_tf)
    test_dataset = datasets.ImageFolder(test_root, test_tf)
    # 护身符：训练/测试类别必须完全一致，否则标签错位、测试结果全废
    if set(train_full.classes) != set(test_dataset.classes) or train_full.classes != test_dataset.classes:
        only_train = sorted(set(train_full.classes) - set(test_dataset.classes))
        only_test = sorted(set(test_dataset.classes) - set(train_full.classes))
        raise SystemExit(
            f'[数据校验失败] 训练与测试类别不一致，已停止（防止白跑）。\n'
            f'训练独有: {only_train}\n测试独有: {only_test}\n'
            f'请检查数据目录: {args.data_path}（常见原因: 混入备份/隐藏文件夹）'
        )
    classes = train_full.classes
    num_classes = len(classes)
    print(f'训练类别数: {len(train_full.classes)} | 测试类别数: {len(test_dataset.classes)} | '
          f'训练样本(全): {len(train_full)} | 测试样本: {len(test_dataset)}')

    # 训练/验证划分（分层）
    idx = np.arange(len(train_full))
    train_idx, val_idx = train_test_split(idx, test_size=args.val_ratio,
                                          stratify=np.array(train_full.targets),
                                          random_state=args.seed)
    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(train_full, val_idx)
    print(f'训练: {len(train_dataset)} | 验证: {len(val_dataset)}')

    # 平衡采样器（基于训练子集）
    t_targets = np.array([train_full.targets[i] for i in train_idx])
    cnt = np.array([np.sum(t_targets == t) for t in np.unique(t_targets)])
    w = torch.from_numpy(1. / cnt[t_targets]).double()
    sampler = WeightedRandomSampler(w, len(w))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8,
                            pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8,
                             pin_memory=True, persistent_workers=True)

    # ---- 模型 ----
    if args.backbone == 'frmt':
        # 消融开关
        multi_branch = args.ablate not in ('reparam', 'baseline')
        use_fgfm = args.ablate in ('full', 'aux', 'clag', 'fgfm')
        use_clag = args.ablate in ('full', 'aux', 'clag')
        use_aux = args.ablate in ('full', 'aux')
        model = FRMobileTransformer(num_classes=num_classes, use_fgfm=use_fgfm, use_clag=use_clag,
                                    use_aux=use_aux, multi_branch=multi_branch,
                                    transformer_layers=args.layers).to(device)
        model_name = 'frmt'
    else:
        model = build_backbone(args.backbone, num_classes).to(device)
        model_name = args.backbone
    n_params = sum(p.numel() for p in model.parameters())
    print(f'模型: {model_name} | 参数量: {n_params/1e6:.2f}M')

    # EMA（评估用）
    ema_avg_fn = get_ema_multi_avg_fn(decay=0.999)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)

    # 优化器：与原代码一致——FRMT 全参 5e-4（原代码'backbone'分组因无匹配参数未生效）；
    # torchvision backbone 则分类头 5e-4、其余 1e-5
    if args.backbone == 'frmt':
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999))
    else:
        head_names = ('classifier', 'fc')
        head = [p for n, p in model.named_parameters() if any(h in n for h in head_names)]
        body = [p for n, p in model.named_parameters() if not any(h in n for h in head_names)]
        optimizer = optim.AdamW([{'params': body, 'lr': 1e-5}, {'params': head, 'lr': 5e-4}],
                                weight_decay=0.05, betas=(0.9, 0.999))
    criterion = CombinedLoss(aux_weight=0.3, label_smoothing=0.1, focal_weight=0.5)
    warmup = WarmupLR(optimizer, warmup_epochs=5, total_epochs=args.epochs,
                      base_lr=5e-4, final_lr=1e-6)

    # ---- 训练 ----
    best_val = 0.0
    best_epoch = 0
    patience = 15
    no_improve = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(args.epochs):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        smooth_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', ncols=110)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            if np.random.random() < 0.5:
                images, la, lb, lam = mixup_data(images, labels, alpha=0.2)
                use_mixup = True
            else:
                use_mixup = False
            optimizer.zero_grad()
            outputs, aux_outputs = forward_model(model, images)
            if use_mixup:
                ce = lam * criterion.ce_loss(outputs, la) + (1 - lam) * criterion.ce_loss(outputs, lb)
                fc = lam * criterion.focal_loss(outputs, la) + (1 - lam) * criterion.focal_loss(outputs, lb)
                loss_main = ce + criterion.focal_weight * fc
                loss_aux = 0.0
                if aux_outputs is not None:
                    loss_aux = lam * criterion.ce_loss(aux_outputs, la) + (1 - lam) * criterion.ce_loss(aux_outputs, lb)
                total = loss_main + criterion.aux_weight * loss_aux
            else:
                total, loss_main, loss_aux = criterion(outputs, aux_outputs, labels)
            total.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema_model.update_parameters(model)
            run_loss += total.item()
            if not use_mixup:
                pred = outputs.argmax(1)
                run_correct += (pred == labels).sum().item()
                run_total += labels.size(0)
            # 平滑 loss（最近 batch 的 EMA），acc 显示为百分比
            smooth_loss = total.item() if smooth_loss == 0.0 else 0.9 * smooth_loss + 0.1 * total.item()
            cur_acc = (run_correct / max(run_total, 1)) * 100 if run_total else 0.0
            pbar.set_postfix(loss=f'{smooth_loss:.4f}', acc=f'{cur_acc:.1f}%', refresh=True)
        train_acc = run_correct / max(run_total, 1)

        # 验证（EMA）
        ema_model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, aux_outputs = forward_model(ema_model, images)
                loss, _, _ = criterion(outputs, aux_outputs, labels)
                v_loss += loss.item()
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += labels.size(0)
        val_acc = v_correct / v_total
        lr_now = warmup.step(epoch)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(run_loss / len(train_loader))
        history['val_loss'].append(v_loss / len(val_loader))
        history['lr'].append(lr_now)
        print(f'Epoch {epoch+1} | Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}% | LR {lr_now:.2e}')
        # 每 5 轮自动更新曲线图（可中途下载查看）
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            save_curves(history, os.path.join(out_dir, '训练曲线.png'))

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({'epoch': epoch, 'model_name': model_name,
                        'model_state_dict': model.state_dict(),
                        'ema_model_state_dict': ema_model.state_dict(),
                        'val_accuracy': val_acc, 'num_classes': num_classes},
                       os.path.join(out_dir, 'best_model_full.pth'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'早停于 Epoch {epoch+1}')
                break

    print(f'训练完成 | 最佳验证准确率 {best_val*100:.2f}% @ epoch {best_epoch+1}')

    # ---- 测试评估（只用一次）----
    ckpt = torch.load(os.path.join(out_dir, 'best_model_full.pth'), weights_only=False)
    ema_model.load_state_dict(ckpt['ema_model_state_dict'])
    ema_model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='测试评估'):
            outputs, _ = forward_model(ema_model, images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    all_probs = np.array(all_probs)
    top1 = (all_preds == all_labels).mean() * 100
    top5 = sum(1 for i in range(len(all_labels)) if all_labels[i] in np.argsort(all_probs[i])[::-1][:5]) / len(all_labels) * 100
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import label_binarize
    yb = label_binarize(all_labels, classes=range(num_classes))
    try:
        auc_roc = roc_auc_score(yb, all_probs, average='macro')
    except Exception:
        auc_roc = float('nan')
    ap = average_precision_score(yb, all_probs, average='macro')
    print(f'Test Top-1 {top1:.2f}% | Top-5 {top5:.2f}% | Macro-F1 {macro_f1:.4f} | mAP {ap:.4f} | AUC {auc_roc:.4f}')

    summary = {'tag': tag, 'model': model_name, 'seed': args.seed, 'ablate': args.ablate,
               'layers': args.layers, 'val_ratio': args.val_ratio,
               'params_m': n_params / 1e6, 'best_val_acc': best_val, 'best_epoch': best_epoch + 1,
               'test_top1': top1, 'test_top5': top5, 'macro_f1': macro_f1, 'map': ap, 'auc_roc': auc_roc,
               'train_samples': len(train_dataset), 'val_samples': len(val_dataset),
               'test_samples': len(test_dataset)}
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f'✅ 完成: {out_dir}')


if __name__ == '__main__':
    main()
