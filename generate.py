import os
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from core.solver import Solver
from core.data_loader import DefaultDataset, CenterCropResize
import core.utils as utils


def load_model(args, device):
    solver = Solver(args)
    solver._load_checkpoint(args.checkpoint)
    nets = solver.nets_ema
    for net in nets.values():
        net.to(device).eval()
    return nets


def run_reference(nets, args, device, transform):
    src_dataset = DefaultDataset(args.src_dir, transform, return_paths=True)
    ref_dataset = ImageFolder(args.ref_dir, transform)

    os.makedirs(args.out_dir, exist_ok=True)

    for src_img, src_name in src_dataset:
        src_img = src_img.unsqueeze(0).to(device)
        idx = 1
        for ref_img, ref_label in ref_dataset:
            ref_img = ref_img.unsqueeze(0).to(device)
            y_ref = torch.tensor([ref_label]).to(device)
            s_ref = nets['style_encoder'](ref_img, y_ref)
            x_fake = nets['generator'](src_img, s_ref, masks=None)
            out_name = f"{Path(src_name).stem}_{idx:02d}.png"
            utils.save_image(x_fake, 1, os.path.join(args.out_dir, out_name))
            idx += 1


def run_latent(nets, args, device, transform):
    src_dataset = DefaultDataset(args.src_dir, transform, return_paths=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for src_img, src_name in src_dataset:
        src_img = src_img.unsqueeze(0).to(device)
        for i in range(args.num_samples):
            y_trg = torch.randint(low=0, high=args.num_domains, size=(1,)).to(device)
            z_trg = torch.randn(1, args.latent_dim).to(device)
            s_trg = nets['mapping_network'](z_trg, y_trg)
            x_fake = nets['generator'](src_img, s_trg, masks=None)
            out_name = f"{Path(src_name).stem}_{i+1:02d}.png"
            utils.save_image(x_fake, 1, os.path.join(args.out_dir, out_name))


def main():
    parser = argparse.ArgumentParser(description="Simple image generation using StarGAN v2 checkpoints")
    parser.add_argument('--expr_dir', type=str, required=True, help='Experiment directory containing checkpoints')
    parser.add_argument('--checkpoint', type=int, default=100000, help='Checkpoint iteration to load')
    parser.add_argument('--src_dir', type=str, required=True, help='Directory of source images')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--mode', type=str, choices=['reference', 'latent'], required=True, help='Generation mode')
    parser.add_argument('--ref_dir', type=str, help='Directory of reference images (required for reference mode)')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of outputs per source for latent mode')
    # model options
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--aspect_ratio', type=float, default=1.0)
    parser.add_argument('--num_domains', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--style_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--w_hpf', type=float, default=1)

    args = parser.parse_args()
    gen_mode = args.mode
    args.mode = 'sample'  # for Solver initialization
    args.checkpoint_dir = os.path.join(args.expr_dir, 'checkpoints')
    args.result_dir = args.out_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets = load_model(args, device)

    height = args.img_size
    width = int(height * args.aspect_ratio)
    transform = transforms.Compose([
        CenterCropResize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if gen_mode == 'reference':
        if not args.ref_dir:
            parser.error('--ref_dir is required for reference mode')
        run_reference(nets, args, device, transform)
    else:
        run_latent(nets, args, device, transform)


if __name__ == '__main__':
    main()
