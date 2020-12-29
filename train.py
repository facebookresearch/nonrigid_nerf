#!/usr/bin/env python3
import warnings
import os
import sys
import shutil
import json
import time
import random

import numpy as np
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from run_nerf_helpers import *
from load_llff import load_llff_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = True  # gets overwritten by args.debug


def batchify(fn, chunk, detailed_output=False):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        if detailed_output:
            outputs, details_lists = zip(
                *[
                    fn(inputs[i : i + chunk], detailed_output=detailed_output)
                    for i in range(0, inputs.shape[0], chunk)
                ]
            )
            outputs = torch.cat(outputs, 0)
            details = {}
            for key in details_lists[0].keys():
                details[key] = torch.cat([details[key] for details in details_lists], 0)
            return outputs, details
        else:
            return torch.cat(
                [
                    fn(inputs[i : i + chunk], detailed_output=detailed_output)
                    for i in range(0, inputs.shape[0], chunk)
                ],
                0,
            )

    return ret


def run_network(
    inputs,
    viewdirs,
    additional_pixel_information,
    fn,
    embed_fn,
    embeddirs_fn,
    netchunk=1024 * 64,
    detailed_output=False,
):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(
        inputs, [-1, inputs.shape[-1]]
    )  # N_rays * N_samples_per_ray x 3
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    ray_bending_latents = additional_pixel_information[
        "ray_bending_latents"
    ]  # N_rays x latent_size
    ray_bending_latents = ray_bending_latents[:, None].expand(
        (inputs.shape[0], inputs.shape[1], ray_bending_latents.shape[-1])
    )  # N_rays x N_samples_per_ray x latent_size
    ray_bending_latents = torch.reshape(
        ray_bending_latents, [-1, ray_bending_latents.shape[-1]]
    )  # N_rays * N_samples_per_ray x latent_size
    embedded = torch.cat(
        [embedded, ray_bending_latents], -1
    )  # N_rays * N_samples_per_ray x (embedded position + embedded viewdirection + latent code)

    outputs_flat = batchify(fn, netchunk, detailed_output)(
        embedded
    )  # fn is model or model_fine from create_nerf(). this calls Nerf.forward(embedded)
    if detailed_output:
        outputs_flat, details = outputs_flat
        for key in details.keys():
            details[key] = torch.reshape(details[key], list(inputs.shape[:-1]) + [-1])
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    if detailed_output:
        return outputs, details
    else:
        return outputs


def batchify_rays(
    rays_flat,
    additional_pixel_information,
    chunk=1024 * 32,
    detailed_output=False,
    **kwargs,
):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        # index correct subset of additional_pixel_information
        relevant_additional_pixel_info = {
            "ray_bending_latents": additional_pixel_information["ray_bending_latents"][
                i : i + chunk, :
            ],
        }

        ret = render_rays(
            rays_flat[i : i + chunk],
            additional_pixel_information=relevant_additional_pixel_info,
            detailed_output=detailed_output,
            **kwargs,
        )
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


class training_wrapper_class(torch.nn.Module):
    def __init__(self, coarse_model, latents, fine_model=None, ray_bender=None):

        super(training_wrapper_class, self).__init__()

        # necessary to duplicate weights correctly across gpus. hacky workaround
        self.coarse_model = coarse_model
        self.latents = latents
        self.fine_model = fine_model
        self.ray_bender = ray_bender
        # self.rgb_mask = nn.Parameter(rgb_mask) # not actual parameters, just a hacky workaround.

    def forward(
        self,
        H,
        W,
        focal,
        ray_params,
        args,
        rays_o,
        rays_d,
        i,
        render_kwargs_train,
        target_s,
        global_step,
        start,
        dataset_extras,
        batch_pixel_indices,
    ):

        # necessary to duplicate weights correctly across gpus. hacky workaround
        self.coarse_model.ray_bender = (self.ray_bender,)
        render_kwargs_train["network_fn"] = self.coarse_model
        render_kwargs_train["ray_bender"] = self.ray_bender
        if self.fine_model is not None:
            self.fine_model.ray_bender = (self.ray_bender,)
            render_kwargs_train["network_fine"] = self.fine_model
        ray_bending_latents_list = self.latents

        ray_bending_latents_list = torch.stack(ray_bending_latents_list, dim=0).to(
            target_s.get_device()
        )  # num_latents x latent_size
        imageid_to_timestepid = torch.tensor(
            dataset_extras["imageid_to_timestepid"]
        )  # num_images

        N_rays = rays_o.shape[0]

        # look up additional information (autodecoder per-image ray bending latent code)
        # need to add this information dynamically here with indexing because otherwise values are not refreshed properly (e.g. if latent codes are concatenated to rays only once at the very start of training)
        additional_pixel_information = {}
        additional_pixel_information["ray_bending_latents"] = ray_bending_latents_list[
            imageid_to_timestepid[batch_pixel_indices[:, 0]], :
        ]  # shape: samples x latent_size

        # regularizers setup
        if args.offsets_loss_weight > 0.0 or args.divergence_loss_weight > 0.0:
            detailed_output = True
        else:
            detailed_output = False

        rgb, disp, acc, extras = render(
            H,
            W,
            focal,
            ray_params,
            rays_o,
            rays_d,
            chunk=args.chunk,
            verbose=i < 10,
            retraw=True,
            additional_pixel_information=additional_pixel_information,
            detailed_output=detailed_output,
            **render_kwargs_train,
        )  # rays need to be split for parallel call

        # consider rgb and ray validity in loss function
        img_loss = img2mse(rgb, target_s, N_rays)
        trans = extras["raw"][..., -1]
        loss = img_loss  # shape: N_rays
        psnr = mse2psnr(img_loss)

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s, N_rays)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # offsets loss
        if self.ray_bender is not None and args.offsets_loss_weight > 0.0:
            offsets = extras["unmasked_offsets"].view(-1, 3)
            weights = extras["visibility_weights"].detach().view(-1)
            # reshape to N_rays x samples and take mean across samples to get shape (N_rays,)
            offsets_loss = torch.mean(
                (weights * torch.norm(offsets, dim=-1)).view(N_rays, -1), dim=-1
            )  # shape: N_rays. L1 loss. "offsets" includes only coarse samples
            offsets_loss += args.rigidity_loss_weight * torch.mean(
                (weights * extras["rigidity_mask"].view(-1)).view(N_rays, -1), dim=-1
            )
            loss = (
                loss
                + args.offsets_loss_weight
                * ((1.0 / 100.0) ** (1 - (global_step / args.N_iters)))
                * offsets_loss
            )  # increasing schedule

        # divergence loss
        if self.ray_bender is not None and args.divergence_loss_weight > 0.0:
            exact_divergence = False
            backprop_into_weights = False
            initial_input_pts = extras["initial_input_pts"].view(
                -1, 3
            )  # num_rays * N_samples x 3
            if "masked_offsets" in extras:
                offsets = extras["masked_offsets"]
            else:
                offsets = extras["unmasked_offsets"]
            offsets = offsets.view(-1, 3)
            weights = extras["opacity_alpha"].view(-1)
            divergence_latents = additional_pixel_information[
                "ray_bending_latents"
            ]  # num_rays x latent_size
            num_rays = divergence_latents.shape[0]
            divergence_latents = (
                divergence_latents.view(num_rays, 1, -1)
                .expand((num_rays, args.N_samples, args.ray_bending_latent_size))
                .reshape(-1, args.ray_bending_latent_size)
            )  # num_rays * N_samples x latent_size

            weights = 1.0 - torch.exp(-F.relu(weights))

            # compute_divergence_loss
            divergence_loss = compute_divergence_loss(
                offsets,
                initial_input_pts,
                divergence_latents,
                render_kwargs_train["ray_bender"],
                exact=exact_divergence,
                chunk=args.chunk,
                N_rays=N_rays,
                weights=weights,
                backprop_into_weights=backprop_into_weights,
            )
            loss = (
                loss
                + args.divergence_loss_weight
                * ((1.0 / 100.0) ** (1 - (global_step / args.N_iters)))
                * divergence_loss
            )  # increasing schedule
        return loss


def get_parallelized_training_function(
    coarse_model, latents, fine_model=None, ray_bender=None
):
    return torch.nn.DataParallel(
        training_wrapper_class(
            coarse_model, latents, fine_model=fine_model, ray_bender=ray_bender
        )
    )


class render_wrapper_class(torch.nn.Module):
    def __init__(self, coarse_model, fine_model=None, ray_bender=None):

        super(render_wrapper_class, self).__init__()

        # hacky workaround to copy network weights to each gpu
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.ray_bender = ray_bender

    def forward(self, *args, **kwargs):
        self.coarse_model.ray_bender = (self.ray_bender,)
        kwargs["network_fn"] = self.coarse_model
        kwargs["ray_bender"] = self.ray_bender
        if self.fine_model is not None:
            self.fine_model.ray_bender = (self.ray_bender,)
            kwargs["network_fine"] = self.fine_model
        return render(*args, **kwargs)


def get_parallelized_render_function(coarse_model, fine_model=None, ray_bender=None):
    return torch.nn.DataParallel(
        render_wrapper_class(coarse_model, fine_model=fine_model, ray_bender=ray_bender)
    )


def render(
    H,
    W,
    focal,
    ray_params,
    rays_o,
    rays_d,
    chunk=1024 * 32,  # c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    additional_pixel_information=None,
    detailed_output=False,
    **kwargs,
):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    device = rays_o[0].get_device()

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            if c2w is None:
                raise RuntimeError(
                    "seems inconsistent, this should only be used for full-image rendering -- need to take care of additional_pixel_information otherwise"
                )
            raise RuntimeError(
                "need to pull this call to get_rays out to render_path() for gpu parallelization to work"
            )
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam, ray_params)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = (
        near * torch.ones_like(rays_d[..., :1], device=device),
        far * torch.ones_like(rays_d[..., :1], device=device),
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(
        rays,
        additional_pixel_information,
        chunk=chunk,
        detailed_output=detailed_output,
        **kwargs,
    )
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(
    render_poses,
    hwf,
    chunk,
    ray_params,
    render_kwargs,
    ray_bending_latents,
    gt_imgs=None,
    savedir=None,
    render_factor=0,
    detailed_output=False,
    parallelized_render_function=None,
):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        if type(focal) == list:
            focal = [focal_i / render_factor for focal_i in focal]
        else:
            focal = focal / render_factor

    rgbs = []
    disps = []
    all_details_and_rest = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        single_latent_code = ray_bending_latents[i]

        this_c2w = c2w[:3, :4]
        device = this_c2w.get_device()
        rays_o, rays_d = get_rays(H, W, focal, this_c2w, ray_params)
        height, width = rays_o.shape[0], rays_o.shape[1]
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        additional_pixel_information = {
            "ray_bending_latents": single_latent_code.reshape(
                1, ray_params["ray_bending_latent_size"]
            ).expand(H * W, ray_params["ray_bending_latent_size"]),
        }

        render_function = (
            render
            if parallelized_render_function is None
            else parallelized_render_function
        )
        rgb, disp, acc, details_and_rest = render_function(
            H,
            W,
            focal,
            ray_params,
            rays_o,
            rays_d,
            chunk=chunk,
            detailed_output=detailed_output,
            additional_pixel_information=additional_pixel_information,
            **render_kwargs,
        )
        rgb = rgb.view(height, width, -1)
        disp = disp.view(height, width)
        acc = acc.view(height, width)
        for key in details_and_rest.keys():
            original_shape = details_and_rest[key].shape
            details_and_rest[key] = (
                details_and_rest[key]
                .view((height, width) + tuple(original_shape[1:]))
                .detach()
                .cpu()
                .numpy()
            )

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if detailed_output:
            all_details_and_rest.append(details_and_rest)
        if i == 0:
            print(rgb.shape, disp.shape)
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

            raw_disparity = disps[-1] / np.max(disps[-1])
            disp8 = to8b(raw_disparity)
            filename = os.path.join(savedir, "disp_{:03d}.png".format(i))
            imageio.imwrite(filename, disp8)

            jet_disp8 = to8b(visualize_disparity_with_jet_color_scheme(raw_disparity))
            filename = os.path.join(savedir, "disp_jet_{:03d}.png".format(i))
            imageio.imwrite(filename, jet_disp8)

            phong_disp8 = to8b(visualize_disparity_with_blinn_phong(raw_disparity))
            filename = os.path.join(savedir, "disp_phong_{:03d}.png".format(i))
            imageio.imwrite(filename, phong_disp8)

            # filename_prefix = os.path.join(savedir, 'ray_bending_{:03d}'.format(i))
            # visualize_ray_bending(details_and_rest["initial_input_pts"], details_and_rest["input_pts"], filename_prefix)

            # if "fine_input_pts" in details_and_rest:
            #    filename_prefix = os.path.join(savedir, 'ray_bending_{:03d}'.format(i))
            #    visualize_ray_bending(details_and_rest["initial_input_pts"].cpu().numpy(), details_and_rest["input_pts"].cpu().numpy(), filename_prefix)

            if gt_imgs is not None:
                try:
                    gt_img = gt_imgs[i].cpu().detach().numpy()
                except:
                    gt_img = gt_imgs[i]
                error = np.linalg.norm(gt_img - rgbs[-1], axis=-1) / np.sqrt(
                    1 + 1 + 1
                )  # height x width
                error *= 10.0  # exaggarate error
                error = np.clip(error, 0.0, 1.0)
                error = to8b(
                    visualize_disparity_with_jet_color_scheme(error)
                )  # height x width x 3. int values in [0,255]
                filename = os.path.join(savedir, "error_{:03d}.png".format(i))
                imageio.imwrite(filename, error)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    if detailed_output:
        return rgbs, disps, all_details_and_rest
    else:
        return rgbs, disps


def create_nerf(args, autodecoder_variables=None, ignore_optimizer=False):
    """Instantiate NeRF's MLP model."""

    grad_vars = []

    if autodecoder_variables is not None:
        grad_vars += autodecoder_variables

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    if args.ray_bending is not None and args.ray_bending != "None":
        ray_bender = ray_bending(
            input_ch, args.ray_bending_latent_size, args.ray_bending, embed_fn
        ).cuda()
        grad_vars += list(ray_bender.parameters())
    else:
        ray_bender = None

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
        if args.approx_nonrigid_viewdirs:
            # netchunk needs to be divisible by both number of samples of coarse and fine Nerfs
            def lcm(x, y):
                from math import gcd

                return x * y // gcd(x, y)

            needs_to_divide = lcm(args.N_samples, args.N_samples + args.N_importance)
            args.netchunk = int(args.netchunk / needs_to_divide) * needs_to_divide
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
        ray_bender=ray_bender,
        ray_bending_latent_size=args.ray_bending_latent_size,
        embeddirs_fn=embeddirs_fn,
        num_ray_samples=args.N_samples,
        approx_nonrigid_viewdirs=args.approx_nonrigid_viewdirs,
    ).cuda()
    grad_vars += list(
        model.parameters()
    )  # model.parameters() does not contain ray_bender parameters

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
            ray_bender=ray_bender,
            ray_bending_latent_size=args.ray_bending_latent_size,
            embeddirs_fn=embeddirs_fn,
            num_ray_samples=args.N_samples + args.N_importance,
            approx_nonrigid_viewdirs=args.approx_nonrigid_viewdirs,
        ).cuda()
        grad_vars += list(model_fine.parameters())

    def network_query_fn(
        inputs,
        viewdirs,
        additional_pixel_information,
        network_fn,
        detailed_output=False,
    ):
        return run_network(
            inputs,
            viewdirs,
            additional_pixel_information,
            network_fn,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk,
            detailed_output=detailed_output,
        )

    # Create optimizer
    # Note: needs to be Adam. otherwise need to check how to avoid wrong DeepSDF-style autodecoder optimization of the per-frame latent codes.
    if ignore_optimizer:
        optimizer = None
    else:
        optimizer = torch.optim.Adam(
            params=grad_vars, lr=args.lrate, betas=(0.9, 0.999)
        )

    start = 0
    logdir = os.path.join(args.rootdir, args.expname, "logs/")
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(logdir, f) for f in sorted(os.listdir(logdir)) if ".tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        if not ignore_optimizer:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])
        if ray_bender is not None:
            ray_bender.load_state_dict(ckpt["ray_bender_state_dict"])
        if autodecoder_variables is not None:
            for latent, saved_latent in zip(
                autodecoder_variables, ckpt["ray_bending_latent_codes"]
            ):
                latent.data[:] = saved_latent[:].detach().clone()

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "ray_bender": ray_bender,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": False,
        "raw_noise_std": args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    # if args.dataset_type != 'llff' or args.no_ndc:
    #    print('Not ndc!')
    render_kwargs_train["ndc"] = False
    render_kwargs_train["lindisp"] = False

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        opacity_color: [num_rays, num_samples]. opacity assigned to each sampled color. independent of ray.
        visibility_weights: [num_rays, num_samples]. Weights assigned to each sampled color. visibility along ray.
        depth_map: [num_rays]. Estimated distance to object.
    """
    device = raw.get_device()

    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise, device=device)

    opacity_alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    visibility_weights = (
        opacity_alpha
        * torch.cumprod(
            torch.cat(
                [
                    torch.ones((opacity_alpha.shape[0], 1), device=device),
                    1.0 - opacity_alpha + 1e-10,
                ],
                -1,
            ),
            -1,
        )[:, :-1]
    )
    rgb_map = torch.sum(visibility_weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(visibility_weights * z_vals, -1)
    acc_map = torch.sum(visibility_weights, -1)
    # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (acc_map + 1e-10))
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map),
        depth_map / torch.sum(visibility_weights, -1),
    )

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, opacity_alpha, visibility_weights, depth_map


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    additional_pixel_information=None,
    detailed_output=False,
    verbose=False,
    pytest=False,
    **dummy_kwargs,
):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand, device=device)

        z_vals = lower + (upper - lower) * t_rand

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples, 3]

    if detailed_output:
        raw, details = network_query_fn(
            pts,
            viewdirs,
            additional_pixel_information,
            network_fn,
            detailed_output=detailed_output,
        )
    else:
        raw = network_query_fn(
            pts,
            viewdirs,
            additional_pixel_information,
            network_fn,
            detailed_output=detailed_output,
        )
    (
        rgb_map,
        disp_map,
        acc_map,
        opacity_alpha,
        visibility_weights,
        depth_map,
    ) = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, opacity_alpha_0, visibility_weights_0 = (
            rgb_map,
            disp_map,
            acc_map,
            opacity_alpha,
            visibility_weights,
        )

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            visibility_weights[..., 1:-1],
            N_importance,
            det=(perturb == 0.0),
            pytest=pytest,
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        if detailed_output:
            raw, fine_details = network_query_fn(
                pts,
                viewdirs,
                additional_pixel_information,
                run_fn,
                detailed_output=detailed_output,
            )
        else:
            raw = network_query_fn(
                pts,
                viewdirs,
                additional_pixel_information,
                run_fn,
                detailed_output=detailed_output,
            )

        (
            rgb_map,
            disp_map,
            acc_map,
            opacity_alpha,
            visibility_weights,
            depth_map,
        ) = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        if detailed_output:
            # N_rays x N_samples_per_ray
            ret["fine_visibility_weights"] = visibility_weights
            # N_rays x N_samples_per_ray
            ret["fine_opacity_alpha"] = opacity_alpha
            for key in fine_details.keys():
                ret["fine_" + str(key)] = fine_details[key]
    if detailed_output:
        # N_rays x N_samples_per_ray
        ret["visibility_weights"] = visibility_weights_0
        ret["opacity_alpha"] = opacity_alpha_0  # N_rays x N_samples_per_ray
        for key in details.keys():
            ret[key] = details[key]

    global DEBUG
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.", flush=True)

    return ret


def config_parser():

    import configargparse

    parser = configargparse.ArgumentParser()
    code_folder = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--config",
        is_config_file=True,
        help="config file path",
        default=os.path.join(code_folder, "configs", "default.txt"),
    )
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--datadir", type=str, help="input data directory")
    parser.add_argument(
        "--rootdir",
        type=str,
        help="root folder where experiment results will be stored: rootdir/expname/",
    )

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_iters", type=int, default=200000, help="number of training iterations"
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250000,
        help="exponential learning rate decay",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument("--seed", type=int, default=-1, help="seeding numpy")
    parser.add_argument(
        "--ray_bending",
        type=str,
        default="None",
        help="which type of ray bending to use (None or simple_neural)",
    )
    parser.add_argument(
        "--ray_bending_latent_size",
        type=int,
        default=32,
        help="size of per-frame autodecoding latent vector used for ray bending",
    )
    parser.add_argument(
        "--approx_nonrigid_viewdirs",
        action="store_true",
        help="approximate nonrigid view directions of the bent ray instead of exact",
    )

    parser.add_argument(
        "--train_block_size",
        type=int,
        default=0,
        help="number of consecutive timesteps to use for training",
    )
    parser.add_argument(
        "--test_block_size",
        type=int,
        default=0,
        help="number of consecutive timesteps to use for testing",
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--offsets_loss_weight",
        type=float,
        default=0.0,
        help="set to 0. for no offsets loss",
    )
    parser.add_argument(
        "--divergence_loss_weight",
        type=float,
        default=0.0,
        help="set to 0. for no divergence loss",
    )
    parser.add_argument(
        "--rigidity_loss_weight",
        type=float,
        default=0.0,
        help="set to 0. for no rigidity loss",
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )

    # training options
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )
    parser.add_argument("--debug", action="store_true", help="enable checking for NaNs")

    # dataset options
    parser.add_argument(
        "--dataset_type", type=str, default="llff", help="options: llff"
    )

    # llff flags
    parser.add_argument(
        "--factor", type=int, default=8, help="downsample factor for LLFF images"
    )
    parser.add_argument(
        "--spherify", action="store_true", help="set for spherical 360 scenes"
    )
    parser.add_argument(
        "--bd_factor",
        type=float,
        default=0.75,
        help="scales the overall scene, NeRF uses 0.75. is ignored.",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=1000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_testset", type=int, default=50000, help="frequency of testset saving"
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=50000,
        help="frequency of render_poses video saving",
    )

    return parser


def _get_multi_view_helper_mappings(num_images):
    imgnames = range(num_images)
    extras = {}
    multi_view_mapping = dict([(name, [i, i]) for i, name in enumerate(imgnames)])

    sorted_multi_view_mapping = {}
    raw_multi_view_list = []
    for key in sorted(multi_view_mapping.keys()):
        sorted_multi_view_mapping[key] = multi_view_mapping[key]
        raw_multi_view_list.append(multi_view_mapping[key])
    extras["raw_multi_view_mapping"] = sorted_multi_view_mapping

    # convert to consecutive numerical ids

    all_timesteps = sorted(
        list(set([timestep for view, timestep in raw_multi_view_list]))
    )
    timestep_to_timestepid = dict(
        [(timestep, i) for i, timestep in enumerate(all_timesteps)]
    )

    all_views = sorted(list(set([view for view, timestep in raw_multi_view_list])))
    view_to_viewid = dict([(view, i) for i, view in enumerate(all_views)])

    extras["raw_timesteps"] = all_timesteps
    extras["rawtimestep_to_timestepid"] = timestep_to_timestepid
    extras["raw_views"] = all_views
    extras["rawview_to_viewid"] = view_to_viewid
    extras["raw_multi_view_list"] = raw_multi_view_list
    extras["imageid_to_viewid"] = [
        view_to_viewid[view] for view, timestep in raw_multi_view_list
    ]
    extras["imageid_to_timestepid"] = [
        timestep_to_timestepid[timestep] for view, timestep in raw_multi_view_list
    ]

    return extras


def main_function(args):

    # miscellaneous initial stuff
    global DEBUG
    DEBUG = args.debug
    torch.autograd.set_detect_anomaly(args.debug)
    if args.seed >= 0:
        np.random.seed(args.seed)

    # camera parameters
    ray_params = {"ray_bending_latent_size": args.ray_bending_latent_size}
    ray_params[
        "focal_x"
    ] = None  # focal_x will be equal to focal_y later on. ray_params allows to set custom values.
    ray_params["focal_y"] = None
    ray_params["center_x"] = None
    ray_params["center_y"] = None

    # Load data

    if args.dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            factor=args.factor,
            recenter=True,
            bd_factor=args.bd_factor,
            spherify=args.spherify,
        )
        dataset_extras = _get_multi_view_helper_mappings(images.shape[0])

        if ray_params["focal_x"] is not None:
            ray_params["focal_x"] /= args.factor
        if ray_params["focal_y"] is not None:
            ray_params["focal_y"] /= args.factor
        if ray_params["center_x"] is not None:
            ray_params["center_x"] /= args.factor
        if ray_params["center_y"] is not None:
            ray_params["center_y"] /= args.factor

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)

        # take out chunks (args parameters: train & test block lengths)
        i_test = []  # [i_test]
        if args.test_block_size > 0 and args.train_block_size > 0:
            print(
                "splitting timesteps into training ("
                + str(args.train_block_size)
                + ") and test ("
                + str(args.test_block_size)
                + ") blocks"
            )
            num_timesteps = len(dataset_extras["raw_timesteps"])
            test_timesteps = np.concatenate(
                [
                    np.arange(
                        min(num_timesteps, blocks_start + args.train_block_size),
                        min(
                            num_timesteps,
                            blocks_start + args.train_block_size + args.test_block_size,
                        ),
                    )
                    for blocks_start in np.arange(
                        0, num_timesteps, args.train_block_size + args.test_block_size
                    )
                ]
            )
            i_test = [
                imageid
                for imageid, timestep in enumerate(
                    dataset_extras["imageid_to_timestepid"]
                )
                if timestep in test_timesteps
            ]

        i_test = np.array(i_test)
        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(int(images.shape[0]))
                if (i not in i_test and i not in i_val)
            ]
        )

        print("DEFINING BOUNDS")
        # if args.no_ndc:
        near = np.ndarray.min(bds) * 0.9
        far = np.ndarray.max(bds) * 1.0
        # else:
        #    near = 0.
        #    far = 1.
        print("NEAR FAR", near, far)

    else:
        print("Unknown dataset type", args.dataset_type, "exiting")
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    if ray_params["focal_x"] is not None:
        focal = [ray_params["focal_x"], ray_params["focal_y"]]
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    logdir = os.path.join(args.rootdir, args.expname, "logs/")
    expname = args.expname
    os.makedirs(logdir, exist_ok=True)
    f = os.path.join(logdir, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(logdir, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # create autodecoder variables as pytorch tensors
    ray_bending_latents_list = [
        torch.zeros(args.ray_bending_latent_size).cuda()
        for _ in range(len(dataset_extras["raw_timesteps"]))
    ]
    for latent in ray_bending_latents_list:
        latent.requires_grad = True

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args, autodecoder_variables=ray_bending_latents_list
    )
    print("start: " + str(start) + " args.N_iters: " + str(args.N_iters), flush=True)

    global_step = start

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    scripts_dict = {"hwf": hwf, "near": near, "far": far}

    coarse_model = render_kwargs_train["network_fn"]
    fine_model = render_kwargs_train["network_fine"]
    ray_bender = render_kwargs_train["ray_bender"]
    parallel_training = get_parallelized_training_function(
        coarse_model=coarse_model,
        latents=ray_bending_latents_list,
        fine_model=fine_model,
        ray_bender=ray_bender,
    )
    parallel_render = get_parallelized_render_function(
        coarse_model=coarse_model, fine_model=fine_model, ray_bender=ray_bender
    )  # only used by render_path() at test time, not for training/optimization

    min_point, max_point = determine_nerf_volume_extent(
        parallel_render, poses, H, W, focal, ray_params, render_kwargs_train, args
    )
    scripts_dict["min_nerf_volume_point"] = min_point.detach().cpu().numpy().tolist()
    scripts_dict["max_nerf_volume_point"] = max_point.detach().cpu().numpy().tolist()

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).cuda()

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # For random ray batching
    print("get rays")
    rays = np.stack(
        [get_rays_np(H, W, focal, p, ray_params) for p in poses[:, :3, :4]], 0
    )  # [N, ro+rd, H, W, 3]
    print("done, concats")

    # attach index information (index among all images in dataset, x and y coordinate)
    image_indices, y_coordinates, x_coordinates = np.meshgrid(
        np.arange(images.shape[0]), np.arange(H), np.arange(W), indexing="ij"
    )  # keep consistent with code in get_rays and get_rays_np. (0,0,0) is coordinate of the top-left corner of the first image, i.e. of [0,0,0]. each array has shape images x height x width
    additional_indices = np.stack(
        [image_indices, x_coordinates, y_coordinates], axis=-1
    )  # N x height x width x 3 (image, x, y)

    rays_rgb = np.concatenate(
        [rays, images[:, None], additional_indices[:, None]], 1
    )  # [N, ro+rd+rgb+ind, H, W, 3]

    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb+ind, 3]

    # use all images
    # keep shape N x H x W x ro+rd+rgb x 3
    rays_rgb = rays_rgb.astype(np.float32)
    print(rays_rgb.shape)

    # Move training data to GPU
    poses = torch.Tensor(poses).cuda()

    # N_iters = 200000 + 1
    N_iters = args.N_iters + 1
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)
    print("Begin", flush=True)

    # Summary writers
    # writer = SummaryWriter(os.path.join(logdir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        optimizer.zero_grad()

        # reset autodecoder gradients to avoid wrong DeepSDF-style optimization. Note: this is only guaranteed to work if the optimizer is Adam
        for latent in ray_bending_latents_list:
            latent.grad = None

        # Sample random ray batch
        # Random over all images
        # use np random to samples N_rand random image IDs, x and y values
        image_indices = np.random.randint(images.shape[0], size=args.N_rand)
        x_coordinates = np.random.randint(W, size=args.N_rand)
        y_coordinates = np.random.randint(H, size=args.N_rand)

        # index rays_rgb with those values
        batch = rays_rgb[
            image_indices, y_coordinates, x_coordinates
        ]  # batch x ro+rd+rgb+ind x 3

        # push to cuda, create batch_rays, target_s, batch_pixel_indices
        batch_pixel_indices = (
            torch.Tensor(
                np.stack([image_indices, x_coordinates, y_coordinates], axis=-1)
            )
            .cuda()
            .long()
        )  # batch x 3
        batch = torch.transpose(torch.Tensor(batch).cuda(), 0, 1)  # 4 x batch x 3
        batch_rays, target_s = batch[:2], batch[2]

        losses = parallel_training(
            H,
            W,
            focal,
            ray_params,
            args,
            batch_rays[0],
            batch_rays[1],
            i,
            render_kwargs_train,
            target_s,
            global_step,
            start,
            dataset_extras,
            batch_pixel_indices,
        )

        # losses will have shape N_rays
        all_test_images_indicator = torch.zeros(images.shape[0], dtype=np.long).cuda()
        all_test_images_indicator[i_test] = 1
        all_training_images_indicator = torch.zeros(
            images.shape[0], dtype=np.long
        ).cuda()
        all_training_images_indicator[i_train] = 1
        # index with image IDs of the N_rays rays to determine weights
        current_test_images_indicator = all_test_images_indicator[
            image_indices
        ]  # N_rays
        current_training_images_indicator = all_training_images_indicator[
            image_indices
        ]  # N_rays

        # first, test_images (if sampled image IDs give non-empty indicators). mask N_rays loss with indicators, then take mean and loss backward with retain_graph=True. then None ray_bender (if existent) and Nerf grads
        if ray_bender is not None and torch.sum(current_test_images_indicator) > 0:
            masked_loss = current_test_images_indicator * losses  # N_rays
            masked_loss = torch.mean(masked_loss)
            masked_loss.backward(retain_graph=True)
            for weights in (
                list(coarse_model.parameters())
                + list([] if fine_model is None else fine_model.parameters())
                + list([] if ray_bender is None else ray_bender.parameters())
            ):
                weights.grad = None
        # next, training images (always). mask N_rays loss with indicators, then take mean and loss backward WITHOUT retain_graph=True
        masked_loss = current_training_images_indicator * losses  # N_rays
        masked_loss = torch.mean(masked_loss)
        masked_loss.backward(retain_graph=False)

        optimizer.step()

        if DEBUG:
            if torch.isnan(losses).any() or torch.isinf(losses).any():
                raise RuntimeError(str(losses))
            if torch.isnan(target_s).any() or torch.isinf(target_s).any():
                raise RuntimeError(str(torch.sum(target_s)) + " " + str(target_s))
            norm_type = 2.0
            total_gradient_norm = 0
            for p in (
                list(coarse_model.parameters())
                + list(fine_model.parameters())
                + list(ray_bender.parameters())
                + list(ray_bending_latents_list)
            ):
                if p.requires_grad and p.grad is not None:
                    param_norm = p.grad.data.norm(norm_type)
                    total_gradient_norm += param_norm.item() ** norm_type
            total_gradient_norm = total_gradient_norm ** (1.0 / norm_type)
            print(total_gradient_norm, flush=True)

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        warming_up = 1000
        if (
            global_step < warming_up
        ):  # in case images are very dark or very bright, need to keep network from initially building up so much momentum that it kills the gradient
            new_lrate /= 20.0 * (-(global_step - warming_up) / warming_up) + 1.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        dt = time.time() - time0
        log_string = (
            "Step: "
            + str(global_step)
            + ", total loss: "
            + str(losses.mean().cpu().detach().numpy())
        )
        if "img_loss0" in locals():
            log_string += ", coarse loss: " + str(
                img_loss0.mean().cpu().detach().numpy()
            )
        if "img_loss" in locals():
            log_string += ", fine loss: " + str(img_loss.mean().cpu().detach().numpy())
        if "offsets_loss" in locals():
            log_string += ", offsets: " + str(
                offsets_loss.mean().cpu().detach().numpy()
            )
        if "divergence_loss" in locals():
            log_string += ", div: " + str(divergence_loss.mean().cpu().detach().numpy())
        log_string += ", time: " + str(dt)
        print(log_string, flush=True)

        # Rest is logging
        if i % args.i_weights == 0:

            all_latents = torch.zeros(0)
            for l in ray_bending_latents_list:
                all_latents = torch.cat([all_latents, l.cpu().unsqueeze(0)], 0)

            if i % 50000 == 0:
                store_extra = True
                path = os.path.join(logdir, "{:06d}.tar".format(i))
            else:
                store_extra = False
                path = os.path.join(logdir, "latest.tar")
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "network_fine_state_dict": None
                    if render_kwargs_train["network_fine"] is None
                    else render_kwargs_train["network_fine"].state_dict(),
                    "ray_bender_state_dict": None
                    if render_kwargs_train["ray_bender"] is None
                    else render_kwargs_train["ray_bender"].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "ray_bending_latent_codes": all_latents,  # shape: frames x latent_size
                    "ray_params": ray_params,
                    "scripts_dict": scripts_dict,
                    "dataset_extras": dataset_extras,
                },
                path,
            )
            del all_latents

            if store_extra:
                shutil.copyfile(path, os.path.join(logdir, "latest.tar"))

            print("Saved checkpoints at", path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            print("rendering test set...", flush=True)
            if len(render_poses) > 0 and len(i_test) > 0:
                with torch.no_grad():
                    if args.render_test:
                        rendering_latents = ray_bending_latents = [
                            ray_bending_latents_list[
                                dataset_extras["imageid_to_timestepid"][i]
                            ]
                            for i in i_test
                        ]
                    else:
                        rendering_latents = ray_bending_latents = [
                            ray_bending_latents_list[
                                dataset_extras["imageid_to_timestepid"][i_test[0]]
                            ]
                            for _ in range(len(render_poses))
                        ]
                    rgbs, disps = render_path(
                        render_poses,
                        hwf,
                        args.chunk,
                        ray_params,
                        render_kwargs_test,
                        ray_bending_latents=rendering_latents,
                        parallelized_render_function=parallel_render,
                    )
                print("Done, saving", rgbs.shape, disps.shape)
                moviebase = os.path.join(logdir, "{}_spiral_{:06d}_".format(expname, i))
                try:
                    imageio.mimwrite(
                        moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8
                    )
                    imageio.mimwrite(
                        moviebase + "disp.mp4",
                        to8b(disps / np.max(disps)),
                        fps=30,
                        quality=8,
                    )
                    imageio.mimwrite(
                        moviebase + "disp_jet.mp4",
                        to8b(
                            np.stack(
                                [
                                    visualize_disparity_with_jet_color_scheme(
                                        disp / np.max(disp)
                                    )
                                    for disp in disps
                                ],
                                axis=0,
                            )
                        ),
                        fps=30,
                        quality=8,
                    )
                    imageio.mimwrite(
                        moviebase + "disp_phong.mp4",
                        to8b(
                            np.stack(
                                [
                                    visualize_disparity_with_blinn_phong(
                                        disp / np.max(disp)
                                    )
                                    for disp in disps
                                ],
                                axis=0,
                            )
                        ),
                        fps=30,
                        quality=8,
                    )
                except:
                    print(
                        "imageio.mimwrite() failed. maybe ffmpeg is not installed properly?"
                    )

            if i >= N_iters + 1 - args.i_video:
                print("rendering full training set...", flush=True)
                with torch.no_grad():
                    rgbs, disps = render_path(
                        poses[i_train],
                        hwf,
                        args.chunk,
                        ray_params,
                        render_kwargs_test,
                        ray_bending_latents=[
                            ray_bending_latents_list[
                                dataset_extras["imageid_to_timestepid"][i]
                            ]
                            for i in i_train
                        ],
                        parallelized_render_function=parallel_render,
                    )
                print("Done, saving", rgbs.shape, disps.shape)
                moviebase = os.path.join(
                    logdir, "{}_training_{:06d}_".format(expname, i)
                )
                try:
                    imageio.mimwrite(
                        moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8
                    )
                    imageio.mimwrite(
                        moviebase + "disp.mp4",
                        to8b(disps / np.max(disps)),
                        fps=30,
                        quality=8,
                    )
                    imageio.mimwrite(
                        moviebase + "disp_jet.mp4",
                        to8b(
                            np.stack(
                                [
                                    visualize_disparity_with_jet_color_scheme(
                                        disp / np.max(disp)
                                    )
                                    for disp in disps
                                ],
                                axis=0,
                            )
                        ),
                        fps=30,
                        quality=8,
                    )
                    imageio.mimwrite(
                        moviebase + "disp_phong.mp4",
                        to8b(
                            np.stack(
                                [
                                    visualize_disparity_with_blinn_phong(
                                        disp / np.max(disp)
                                    )
                                    for disp in disps
                                ],
                                axis=0,
                            )
                        ),
                        fps=30,
                        quality=8,
                    )
                except:
                    print(
                        "imageio.mimwrite() failed. maybe ffmpeg is not installed properly?"
                    )

        if i % args.i_testset == 0 and i > 0:
            trainsubsavedir = os.path.join(logdir, "trainsubset_{:06d}".format(i))
            os.makedirs(trainsubsavedir, exist_ok=True)
            i_train_sub = i_train
            if i >= N_iters + 1 - args.i_video:
                i_train_sub = i_train_sub
            else:
                i_train_sub = i_train_sub[
                    :: np.maximum(1, int((len(i_train_sub) / len(i_test)) + 0.5))
                ]
            print("i_train_sub poses shape", poses[i_train_sub].shape)
            with torch.no_grad():
                render_path(
                    poses[i_train_sub],
                    hwf,
                    args.chunk,
                    ray_params,
                    render_kwargs_test,
                    gt_imgs=images[i_train_sub],
                    savedir=trainsubsavedir,
                    detailed_output=True,
                    ray_bending_latents=[
                        ray_bending_latents_list[
                            dataset_extras["imageid_to_timestepid"][i]
                        ]
                        for i in i_train_sub
                    ],
                    parallelized_render_function=parallel_render,
                )
            print("Saved some training images")

            if len(i_test) > 0:
                testsavedir = os.path.join(logdir, "testset_{:06d}".format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print("test poses shape", poses[i_test].shape)
                with torch.no_grad():
                    render_path(
                        poses[i_test],
                        hwf,
                        args.chunk,
                        ray_params,
                        render_kwargs_test,
                        gt_imgs=images[i_test],
                        savedir=testsavedir,
                        detailed_output=True,
                        ray_bending_latents=[
                            ray_bending_latents_list[
                                dataset_extras["imageid_to_timestepid"][i]
                            ]
                            for i in i_test
                        ],
                        parallelized_render_function=parallel_render,
                    )
                print("Saved test set")

        if i % args.i_print == 0:
            if "psnr" in locals():
                tqdm.write(
                    f"[TRAIN] Iter: {i} Loss: {losses.mean().item()}  PSNR: {psnr.item()}"
                )
            else:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {losses.mean().item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1
        print("", end="", flush=True)


def create_folder(folder):
    import pathlib

    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


def backup(results_folder):
    print("backing up... ", flush=True, end="")
    special_files_to_copy = ["configs/default.txt"]  # ["specs.json"]
    filetypes_to_copy = [".py"]
    subfolders_to_copy = ["", "llff_preprocessing/"]

    this_file = os.path.realpath(__file__)
    this_folder = os.path.dirname(this_file) + "/"
    backup_folder = os.path.join(results_folder, "backup/")
    create_folder(backup_folder)
    # special files
    [
        create_folder(os.path.join(backup_folder, os.path.split(file)[0]))
        for file in special_files_to_copy
    ]
    [
        shutil.copyfile(
            os.path.join(this_folder, file), os.path.join(backup_folder, file)
        )
        for file in special_files_to_copy
    ]
    # folders
    for subfolder in subfolders_to_copy:
        create_folder(os.path.join(backup_folder, subfolder))
        files = os.listdir(os.path.join(this_folder, subfolder))
        files = [
            file
            for file in files
            if os.path.isfile(os.path.join(this_folder, subfolder, file))
            and file[file.rfind(".") :] in filetypes_to_copy
        ]
        [
            shutil.copyfile(
                os.path.join(this_folder, subfolder, file),
                os.path.join(backup_folder, subfolder, file),
            )
            for file in files
        ]

    print("done.", flush=True)


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    results_folder = os.path.join(args.rootdir, args.expname + "/")
    print(results_folder, flush=True)

    create_folder(results_folder)
    if args.no_reload:
        shutil.rmtree(results_folder)
    backup(results_folder)

    main_function(args)
