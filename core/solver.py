"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from collections import defaultdict
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None
            masks = None

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def generate(self, loaders, args):
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        for net in nets_ema.values():
            net.eval()

        src_loader = loaders.src
        class_to_idx = getattr(src_loader.dataset, 'class_to_idx', {})
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        if hasattr(loaders, 'ref'):
            ref_class_to_idx = getattr(loaders.ref.dataset, 'class_to_idx', {})
            for name, idx in ref_class_to_idx.items():
                idx_to_class.setdefault(idx, name)

        def _parse_target(target):
            if target is None:
                return list(range(args.num_domains))
            if isinstance(target, str):
                token = target.strip()
                if token in idx_to_class.values():
                    indices = [idx for idx, name in idx_to_class.items() if name == token]
                    if indices:
                        return indices
                    raise ValueError('Unknown target_domain: %s' % target)
                if token.isdigit():
                    idx = int(token)
                else:
                    raise ValueError('Unknown target_domain: %s' % target)
            else:
                idx = int(target)
            if idx < 0 or idx >= args.num_domains:
                raise ValueError('target_domain %s is out of range [0, %d).' % (target, args.num_domains))
            return [idx]

        target_domains = sorted(set(_parse_target(args.target_domain)))
        if not target_domains:
            raise ValueError('No target domains available for generation.')

        use_references = hasattr(loaders, 'ref')
        style_cache = {idx: [] for idx in target_domains}

        if args.num_samples < 1:
            raise ValueError('num_samples must be a positive integer.')

        if use_references:
            ref_loader = loaders.ref
            for x_ref, y_ref, _ in ref_loader:
                x_ref = x_ref.to(self.device)
                y_ref = y_ref.to(self.device)
                styles = nets_ema.style_encoder(x_ref, y_ref)
                for style, label in zip(styles, y_ref):
                    label_idx = int(label.item())
                    if label_idx in style_cache and len(style_cache[label_idx]) < args.num_samples:
                        style_cache[label_idx].append(style.detach().cpu())
                if all(len(style_cache[idx]) >= args.num_samples for idx in target_domains):
                    break

            available_domains = [idx for idx in target_domains if len(style_cache[idx]) > 0]
            missing_domains = [idx for idx in target_domains if len(style_cache[idx]) == 0]
            if missing_domains:
                domain_list = ', '.join(idx_to_class.get(idx, str(idx)) for idx in missing_domains)
                if len(available_domains) == 0:
                    print('No reference images found for %s. Falling back to latent sampling.' % domain_list)
                    use_references = False
                else:
                    print('Skipping domains without references: %s' % domain_list)
                    target_domains = available_domains

        if not use_references:
            print('Generating samples using latent codes for domains: {}'
                  .format(', '.join(idx_to_class.get(idx, str(idx)) for idx in target_domains)))

        for net in self.nets.values():
            net.eval()

        counters = defaultdict(int)

        for x_src, _, names in src_loader:
            x_src = x_src.to(self.device)
            batch_size = x_src.size(0)

            for domain_idx in target_domains:
                y_trg = torch.full((batch_size,), domain_idx, dtype=torch.long, device=self.device)
                for sample_idx in range(args.num_samples):
                    if use_references:
                        styles = style_cache[domain_idx]
                        if not styles:
                            continue
                        style_code = styles[sample_idx % len(styles)].to(self.device)
                        style_code = style_code.unsqueeze(0).repeat(batch_size, 1)
                    else:
                        z_trg = torch.randn(batch_size, args.latent_dim).to(self.device)
                        style_code = nets_ema.mapping_network(z_trg, y_trg)

                    x_fake = nets_ema.generator(x_src, style_code, masks=None)

                    for i, name in enumerate(names):
                        base_name = str(name)
                        counters[base_name] += 1
                        outfile = ospj(args.result_dir, f'{base_name}_{counters[base_name]:02d}.png')
                        utils.save_image(x_fake[i:i+1], 1, outfile)
                        print('Saved {}'.format(outfile))

    @torch.no_grad()
    def sample_with_latent(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))

        # latent-guided translation with fixed domain and random latent
        fname = ospj(args.result_dir, 'latent.jpg')
        print('Working on {}...'.format(fname))

        # target domain: use domain 1 (e.g., 'male' for CelebA)
        y_trg_list = [torch.ones(src.x.size(0)).long().to(self.device)]

        # latent vectors: generate multiple samples for style variation
        z_trg_list = [torch.randn(src.x.size(0), args.latent_dim).to(self.device)
                      for _ in range(args.num_outs_per_domain)]

        # psi controls style strength (0 = avg style, 1 = full style)
        utils.translate_using_latent(nets_ema, args, src.x, y_trg_list, z_trg_list, psi=1.0, filename=fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    # masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    masks = None

    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

