# -*- coding: utf-8 -*-
"""FR-MobileTransformer 评估脚本（只用已训模型，无需重训）
====================================================
功能：
  [1] 鲁棒性：高斯噪声 σ=0~0.30 曲线 + 高斯模糊/亮度变化/JPEG 压缩三类扰动
      （对比曲线需先训练 baseline 对比模型：exp_frmt_experiments.py --backbone efficientnet_b0）
  [2] t-SNE 特征分布可视化
  [3] Grad-CAM 热力图（作用于最后一个卷积层）
  [4] 效率实测：参数量 / FLOPs / 模型大小 / FPS

用法：
  python exp_eval.py --ckpt_dir experiments/full_s42 --data_path /root/autodl-tmp/project/data_clean --mode all
  python exp_eval.py --ckpt_dir experiments/full_s42 --data_path ... --mode robust
  python exp_eval.py --ckpt_dir experiments/full_s42 --data_path ... --mode efficiency
输出：eval_out/<tag>/
"""
import os, argparse, json, time, csv, io
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import exp_frmt_experiments as E


def _strip_module_prefix(state_dict):
    """兼容 DataParallel 保存的权重：去掉 'module.' 前缀、忽略 EMA 计数器（n_averaged）。"""
    fixed = {}
    for k, v in state_dict.items():
        if k == 'n_averaged':
            continue
        if k.startswith('module.'):
            k = k[len('module.'):]
        fixed[k] = v
    return fixed


def _model_cfg_from_summary(summary):
    """根据 summary 中的 ablate 字段重建与训练一致的模型结构。"""
    model_name = summary.get('model', 'frmt')
    if model_name != 'frmt':
        return {'model_name': model_name}
    ablate = summary.get('ablate', 'full')
    return {
        'model_name': 'frmt',
        'use_fgfm': ablate in ('full', 'aux', 'clag', 'fgfm'),
        'use_clag': ablate in ('full', 'aux', 'clag'),
        'use_aux': ablate in ('full', 'aux'),
        'multi_branch': ablate not in ('reparam', 'baseline'),
        'transformer_layers': summary.get('layers', 1),
    }


def load_model(ckpt_dir, device):
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model_full.pth'), weights_only=False)
    with open(os.path.join(ckpt_dir, 'summary.json'), encoding='utf-8') as f:
        summary = json.load(f)
    cfg = _model_cfg_from_summary(summary)
    num_classes = summary.get('num_classes', 39)
    if cfg['model_name'] == 'frmt':
        model = E.FRMobileTransformer(num_classes=num_classes,
                                      use_fgfm=cfg['use_fgfm'], use_clag=cfg['use_clag'],
                                      use_aux=cfg['use_aux'], multi_branch=cfg['multi_branch'],
                                      transformer_layers=cfg['transformer_layers'])
    else:
        model = E.build_backbone(cfg['model_name'], num_classes)
    state = ckpt['ema_model_state_dict'] if 'ema_model_state_dict' in ckpt else ckpt['model_state_dict']
    model.load_state_dict(_strip_module_prefix(state))
    model.to(device).eval()
    return model, cfg['model_name'], ckpt, summary


def model_forward(model, model_name, x, return_features=False):
    """统一 FRMT（返回 output, aux, [features]）与 torchvision backbone（仅返回 logits）的调用。"""
    if model_name == 'frmt':
        return model(x, return_features=return_features)
    out = model(x)
    if return_features:
        return out, None, out  # backbone 无显式特征接口，t-SNE 降级用 logits
    return out, None


def get_test_loader(data_path):
    _, test_tf = E.build_transforms()
    ds = datasets.ImageFolder(os.path.join(data_path, 'test'), test_tf)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True), ds


def eval_accuracy(model, model_name, loader, device, perturb=None):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if perturb is not None:
                images = perturb(images)
            outputs, _ = model_forward(model, model_name, images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100


def jpeg_q(im, q):
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# ============================================================
# 1. 鲁棒性
# ============================================================
def run_robust(model, model_name, loader, device, tag, out_dir, data_path):
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    # --- 高斯噪声曲线 ---
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    accs = []
    for std in noise_levels:
        acc = eval_accuracy(model, model_name, loader, device,
                            perturb=(lambda imgs, s=std: imgs if s == 0 else imgs + torch.randn_like(imgs) * s))
        accs.append(acc)
        rows.append({'type': 'gaussian_noise', 'sigma': std, 'acc': round(acc, 2)})
        print(f'噪声 σ={std}: {acc:.2f}%')

    plt.figure(figsize=(7, 5))
    plt.plot(noise_levels, accs, 'o-', color='#D55E00', lw=2.5, ms=8, label=tag)
    plt.xlabel('Gaussian Noise σ'); plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105); plt.grid(alpha=0.3, ls='--')
    plt.title('Robustness under Gaussian Noise'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'robust_noise_{tag}.png'), dpi=300)
    plt.close()

    # --- 三类扰动（PIL 层面：扰动原图 → Resize/Crop → Norm）---
    base = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean, std)
    raw_full = datasets.ImageFolder(os.path.join(data_path, 'test'))
    order = np.random.RandomState(0).permutation(len(raw_full)).tolist()
    raw_ds = Subset(raw_full, order)
    base_ds, indices = raw_ds.dataset, list(raw_ds.indices)
    paths = [base_ds.samples[i][0] for i in indices]
    labels = [base_ds.samples[i][1] for i in indices]

    def acc_with(fn):
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                imgs, labs = [], []
                for p in paths[i:i+64]:
                    img = base(fn(Image.open(p).convert('RGB')))
                    t = norm(torch.from_numpy(np.array(img).astype(np.float32) / 255.).permute(2, 0, 1))
                    imgs.append(t)
                labs = torch.tensor(labels[i:i+64]).to(device)
                outputs, _ = model_forward(model, model_name, torch.stack(imgs).to(device))
                correct += (outputs.argmax(1) == labs).sum().item()
                total += len(labs)
        return correct / total * 100

    for name, fn in [
        ('gaussian_blur', lambda im: im.filter(ImageFilter.GaussianBlur(radius=2))),
        ('brightness_0.6', lambda im: ImageEnhance.Brightness(im).enhance(0.6)),
        ('brightness_1.5', lambda im: ImageEnhance.Brightness(im).enhance(1.5)),
        ('jpeg_q50', lambda im: jpeg_q(im, 50)),
        ('jpeg_q70', lambda im: jpeg_q(im, 70)),
    ]:
        acc = acc_with(fn)
        rows.append({'type': name, 'sigma': '', 'acc': round(acc, 2)})
        print(f'{name}: {acc:.2f}%')

    with open(os.path.join(out_dir, f'robustness_{tag}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['type', 'sigma', 'acc'])
        w.writeheader(); w.writerows(rows)


# ============================================================
# 2. t-SNE
# ============================================================
def run_tsne(model, model_name, loader, classes, tag, out_dir, device, n_samples=3000):
    feats, labs = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            _, _, feat = model_forward(model, model_name, imgs.to(device), return_features=True)
            feats.append(feat.cpu().numpy()); labs.extend(lbls.numpy())
            if len(labs) >= n_samples:
                break
    feats = np.concatenate(feats)[:n_samples]
    labs = np.array(labs)[:n_samples]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb = tsne.fit_transform(feats)
    plt.figure(figsize=(9, 8), dpi=300)
    top = np.bincount(labs).argsort()[-10:]
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#7f7f7f']
    bg = ~np.isin(labs, top)
    plt.scatter(emb[bg, 0], emb[bg, 1], c='#E0E0E0', s=10, alpha=0.4, label='other classes')
    for i, cls in enumerate(top):
        m = labs == cls
        name = classes[cls].split('___')[-1][:18].replace('_', ' ')
        plt.scatter(emb[m, 0], emb[m, 1], c=palette[i % 10], s=20, alpha=0.8, edgecolor='white', lw=0.4, label=name)
    plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'tsne_{tag}.png'), dpi=300)
    plt.close()
    print('t-SNE 已保存')


# ============================================================
# 3. Grad-CAM（作用于最后一个卷积层）
# ============================================================
def find_last_conv(model):
    conv = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv = m
    return conv


def run_gradcam(model, model_name, loader, classes, tag, out_dir, device, n=4):
    conv = find_last_conv(model)
    if conv is None:
        print('未找到卷积层，跳过 Grad-CAM'); return
    acts, grads = {}, {}

    def fa(m, i, o): acts['v'] = o
    def fb(m, i, o): grads['v'] = o[0]
    h1 = conv.register_forward_hook(fa)
    h2 = conv.register_full_backward_hook(fb)

    model.eval()
    count = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        imgs.requires_grad_(True)
        model.zero_grad()
        outputs, _ = model_forward(model, model_name, imgs)
        idx = outputs.argmax(1)
        onehot = torch.zeros_like(outputs)
        for i in range(imgs.size(0)):
            onehot[i, idx[i]] = 1
        outputs.backward(gradient=onehot)
        weight = grads['v'].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weight * acts['v']).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze(1).detach().cpu().numpy()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (imgs.cpu().detach() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        for i in range(imgs.size(0)):
            if count >= n:
                break
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(img[i]); ax[0].set_title(classes[lbls[i]].split('___')[-1]); ax[0].axis('off')
            ax[1].imshow(cam[i], cmap='jet'); ax[1].set_title('Grad-CAM'); ax[1].axis('off')
            ax[2].imshow(img[i]); ax[2].imshow(cam[i], cmap='jet', alpha=0.5); ax[2].set_title('Overlay'); ax[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'gradcam_{tag}_{count}.png'), dpi=300)
            plt.close()
            count += 1
        if count >= n:
            break
    h1.remove(); h2.remove()
    print(f'Grad-CAM 已保存 {count} 张')


# ============================================================
# 4. 效率
# ============================================================
def run_efficiency(model, model_name, loader, ckpt_path, tag, out_dir, device):
    n_params = sum(p.numel() for p in model.parameters())
    size_kb = os.path.getsize(ckpt_path) / 1024
    flops_g = None
    try:
        from thop import profile
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        flops, _ = profile(model, inputs=(dummy,))
        flops_g = flops / 1e9
    except Exception as e:
        print(f'thop 不可用（pip install thop 后重跑）: {e}')
    model.eval()
    n_warm, n_run = 5, 30
    t0 = None
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            model_forward(model, model_name, imgs.to(device))
            if i == n_warm:
                t0 = time.time()
            if t0 is not None and i >= n_warm + n_run:
                break
    fps = (n_run * 64) / (time.time() - t0) if t0 else 0.0
    row = {'params_m': round(n_params / 1e6, 3), 'flops_g': round(flops_g, 3) if flops_g else 'TBD',
           'model_size_kb': round(size_kb, 1), 'fps_gpu': round(fps, 1)}
    print(row)
    with open(os.path.join(out_dir, f'efficiency_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(row, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--mode', default='all', choices=['all', 'robust', 'tsne', 'gradcam', 'efficiency'])
    parser.add_argument('--num_samples', type=int, default=4, help='Grad-CAM 样本数')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_name, ckpt, summary = load_model(args.ckpt_dir, device)
    tag = model_name
    out_dir = os.path.join('eval_out', os.path.basename(args.ckpt_dir.rstrip('/\\')))
    os.makedirs(out_dir, exist_ok=True)
    loader, ds = get_test_loader(args.data_path)
    classes = ds.classes

    if args.mode in ('all', 'robust'):
        run_robust(model, model_name, loader, device, tag, out_dir, args.data_path)
    if args.mode in ('all', 'tsne'):
        run_tsne(model, model_name, loader, classes, tag, out_dir, device)
    if args.mode in ('all', 'gradcam'):
        run_gradcam(model, model_name, loader, classes, tag, out_dir, device, n=args.num_samples)
    if args.mode in ('all', 'efficiency'):
        run_efficiency(model, model_name, loader, os.path.join(args.ckpt_dir, 'best_model_full.pth'),
                       tag, out_dir, device)
    print('✅ 评估完成:', out_dir)


if __name__ == '__main__':
    main()
