# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import math
import shutil
import os
import pathlib
import re
import sys

sys.stdout.flush()
from PIL import Image
import imageio
import torch


def _setup_nonrigid_nerf_network(results_folder, checkpoint="latest"):

    extra_sys_folder = os.path.join(results_folder, "backup/")

    import sys

    sys.path.append(extra_sys_folder)

    from train import (
        config_parser,
        create_nerf,
        render_path,
        get_parallelized_render_function,
        _get_multi_view_helper_mappings,
    )

    args = config_parser().parse_args(
        ["--config", os.path.join(results_folder, "logs", "args.txt")]
    )

    print(args, flush=True)

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args, autodecoder_variables=None, ignore_optimizer=True
    )

    def load_weights_into_network(render_kwargs_train, checkpoint=None, path=None):
        if path is not None and checkpoint is not None:
            raise RuntimeError("trying to load weights from two sources")
        if checkpoint is not None:
            path = os.path.join(results_folder, "logs", checkpoint + ".tar")
        checkpoint_dict = torch.load(path)
        start = checkpoint_dict["global_step"]
        # optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        render_kwargs_train["network_fn"].load_state_dict(
            checkpoint_dict["network_fn_state_dict"]
        )
        if render_kwargs_train["network_fine"] is not None:
            render_kwargs_train["network_fine"].load_state_dict(
                checkpoint_dict["network_fine_state_dict"]
            )
        if render_kwargs_train["ray_bender"] is not None:
            render_kwargs_train["ray_bender"].load_state_dict(
                checkpoint_dict["ray_bender_state_dict"]
            )
        return checkpoint_dict

    checkpoint_dict = load_weights_into_network(
        render_kwargs_train, checkpoint=checkpoint
    )

    def get_training_ray_bending_latents(checkpoint="latest"):
        training_latent_vectors = os.path.join(
            results_folder, "logs", checkpoint + ".tar"
        )
        training_latent_vectors = torch.load(training_latent_vectors)[
            "ray_bending_latent_codes"
        ]
        return training_latent_vectors  # shape: frames x latent_size

    from run_nerf_helpers import determine_nerf_volume_extent
    from load_llff import load_llff_data

    def load_llff_dataset(
        render_kwargs_train_=None,
        render_kwargs_test_=None,
        return_nerf_volume_extent=False,
    ):

        datadir = args.datadir
        factor = args.factor
        spherify = args.spherify
        bd_factor = args.bd_factor

        # actual loading
        images, poses, bds, render_poses, i_test = load_llff_data(
            datadir,
            factor=factor,
            recenter=True,
            bd_factor=bd_factor,
            spherify=spherify,
        )
        extras = _get_multi_view_helper_mappings(images.shape[0])

        # poses
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]  # N x 3 x 4
        all_rotations = poses[:, :3, :3]  # N x 3 x 3
        all_translations = poses[:, :3, 3]  # N x 3

        render_poses = render_poses[:, :3, :4]
        render_rotations = render_poses[:, :3, :3]
        render_translations = render_poses[:, :3, 3]

        # splits
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

        # near, far
        # if args.no_ndc:
        near = np.ndarray.min(bds) * 0.9
        far = np.ndarray.max(bds) * 1.0
        # else:
        #    near = 0.
        #    far = 1.
        bds_dict = {
            "near": near,
            "far": far,
        }
        if render_kwargs_train_ is not None:
            render_kwargs_train_.update(bds_dict)
        if render_kwargs_test_ is not None:
            render_kwargs_test_.update(bds_dict)

        if return_nerf_volume_extent:
            ray_params = checkpoint_dict["ray_params"]
            min_point, max_point = determine_nerf_volume_extent(
                get_parallelized_render_function(),
                poses,
                hwf[0],
                hwf[1],
                hwf[2],
                ray_params,
                render_kwargs_test,
            )
            extras["min_nerf_volume_point"] = min_point.detach()
            extras["max_nerf_volume_point"] = max_point.detach()

        return (
            images,
            poses,
            all_rotations,
            all_translations,
            bds,
            render_poses,
            render_rotations,
            render_translations,
            i_train,
            i_val,
            i_test,
            near,
            far,
            extras,
        )

    raw_render_path = render_path

    def render_convenient(
        rotations=None,
        translations=None,
        poses=None,
        detailed_output=None,
        ray_bending_latents=None,
        render_factor=None,
        with_ray_bending=None,
        custom_checkpoint_dict=None,
        hwf=None,
        chunk=None,
        custom_ray_params=None,
        custom_render_kwargs_test=None,
        rigidity_test_time_cutoff=None,
    ):

        # poses should have shape Nx3x4, rotations Nx3x3, translations Nx3 (or Nx3x1 or Nx1x3 or 3)
        # ray_bending_latents are a list of latent codes or an array of shape N x latent_size

        # poses
        if poses is None:
            if rotations is None or translations is None:
                raise RuntimeError
            rotations = torch.Tensor(rotations).reshape(-1, 3, 3)
            translations = torch.Tensor(translations).reshape(-1, 3, 1)
            poses = torch.cat([rotations, translations], -1)  # N x 3 x 4
        else:
            if rotations is not None or translations is not None:
                raise RuntimeError
        if len(poses.shape) > 3:
            raise RuntimeError
        if (
            poses.shape[-1] == 5
        ):  # the standard poses that are loaded by load_llff have hwf in the last column, but that's ignored anyway later on, so throw away here for simplicity
            poses = poses[..., :4]
        poses = torch.Tensor(poses).cuda().reshape(-1, 3, 4)

        # other parameters/arguments
        checkpoint_dict_ = (
            checkpoint_dict
            if custom_checkpoint_dict is None
            else custom_checkpoint_dict
        )
        ray_params_ = (
            checkpoint_dict_["ray_params"]
            if custom_ray_params is None
            else custom_ray_params
        )
        render_kwargs_test_ = (
            render_kwargs_test
            if custom_render_kwargs_test is None
            else custom_render_kwargs_test
        )
        if hwf is None:
            hwf = checkpoint_dict_["scripts_dict"]["hwf"]
        if chunk is None:
            chunk = args.chunk
        if render_factor is None:
            render_factor = 0
        if detailed_output is None:
            detailed_output = False
        if with_ray_bending is None:
            with_ray_bending = True

        if with_ray_bending:
            backup_rigidity_test_time_cutoff = render_kwargs_test_[
                "ray_bender"
            ].rigidity_test_time_cutoff
            render_kwargs_test_[
                "ray_bender"
            ].rigidity_test_time_cutoff = rigidity_test_time_cutoff
        else:
            backup_ray_bender = render_kwargs_test_["network_fn"].ray_bender
            render_kwargs_test_["network_fn"].ray_bender = (None,)
            render_kwargs_test_["ray_bender"] = None
            if "network_fine" in render_kwargs_test_:
                render_kwargs_test_["network_fine"].ray_bender = (None,)

        coarse_model = render_kwargs_test_["network_fn"]
        fine_model = render_kwargs_test_["network_fine"]
        ray_bender = render_kwargs_test_["ray_bender"]
        parallel_render = get_parallelized_render_function(
            coarse_model=coarse_model, fine_model=fine_model, ray_bender=ray_bender
        )
        with torch.no_grad():
            returned_outputs = render_path(
                poses,
                hwf,
                args.chunk,
                ray_params_,
                render_kwargs_test_,
                render_factor=render_factor,
                detailed_output=detailed_output,
                ray_bending_latents=ray_bending_latents,
                parallelized_render_function=parallel_render,
            )

        if with_ray_bending:
            render_kwargs_test_[
                "ray_bender"
            ].rigidity_test_time_cutoff = backup_rigidity_test_time_cutoff
        else:
            render_kwargs_test_["network_fn"].ray_bender = backup_ray_bender
            render_kwargs_test_["ray_bender"] = backup_ray_bender[0]
            if "network_fine" in render_kwargs_test_:
                render_kwargs_test_["network_fine"].ray_bender = backup_ray_bender

        if detailed_output:
            rgbs, disps, details_and_rest = returned_outputs
            return (
                rgbs,
                disps,
                details_and_rest,
            )  # N x height x width x 3, N x height x width. RGB values in [0,1]
        else:
            rgbs, disps = returned_outputs
            return (
                rgbs,
                disps,
            )  # N x height x width x 3, N x height x width. RGB values in [0,1]

    from run_nerf_helpers import (
        to8b,
        visualize_disparity_with_jet_color_scheme,
        visualize_disparity_with_blinn_phong,
        visualize_ray_bending,
    )

    def convert_rgb_to_saveable(rgb):
        # input: float values in [0,1]
        # output: int values in [0,255]
        return to8b(rgb)

    def convert_disparity_to_saveable(disparity, normalize=True):
        # takes in a single disparity map of shape height x width.
        # can be saved via: imageio.imwrite(filename, convert_disparity_to_saveable(disparity))
        converted_disparity = (
            disparity / np.max(disparity) if normalize else disparity.copy()
        )
        converted_disparity = to8b(
            converted_disparity
        )  # height x width. int values in [0,255].
        return converted_disparity

    def convert_disparity_to_jet(disparity, normalize=True):
        converted_disparity = (
            disparity / np.max(disparity) if normalize else disparity.copy()
        )
        converted_disparity = to8b(
            visualize_disparity_with_jet_color_scheme(converted_disparity)
        )
        return converted_disparity  # height x width x 3. int values in [0,255].

    def convert_disparity_to_phong(disparity, normalize=True):
        converted_disparity = (
            disparity / np.max(disparity) if normalize else disparity.copy()
        )
        converted_disparity = to8b(
            visualize_disparity_with_blinn_phong(converted_disparity)
        )
        return converted_disparity  # height x width x 3. int values in [0,255].

    def store_ray_bending_mesh_visualization(
        initial_input_pts, input_pts, filename_prefix, subsampled_target=None
    ):
        # initial_input_pts: rays x samples_per_ray x 3
        # input_pts: rays x samples_per_ray x 3
        return visualize_ray_bending(
            initial_input_pts,
            input_pts,
            filename_prefix,
            subsampled_target=subsampled_target,
        )

    sys.path.remove(extra_sys_folder)

    return (
        render_kwargs_train,
        render_kwargs_test,
        start,
        grad_vars,
        load_weights_into_network,
        checkpoint_dict,
        get_training_ray_bending_latents,
        load_llff_dataset,
        raw_render_path,
        render_convenient,
        convert_rgb_to_saveable,
        convert_disparity_to_saveable,
        convert_disparity_to_jet,
        convert_disparity_to_phong,
        store_ray_bending_mesh_visualization,
        to8b,
    )


def create_folder(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


def free_viewpoint_rendering(args):

    # determine output name
    if args.camera_path == "spiral":
        output_name = args.deformations + "_" + args.camera_path
    elif args.camera_path == "fixed":
        output_name = (
            args.deformations + "_" + args.camera_path + "_" + str(args.fixed_view)
        )
    elif args.camera_path == "input_recontruction":
        output_name = args.deformations + "_" + args.camera_path
    else:
        raise RuntimeError("invalid --camera_path argument")

    output_folder = os.path.join(args.input, "output", output_name)
    create_folder(output_folder)

    # load Nerf network
    (
        render_kwargs_train,
        render_kwargs_test,
        start,
        grad_vars,
        load_weights_into_network,
        checkpoint_dict,
        get_training_ray_bending_latents,
        load_llff_dataset,
        raw_render_path,
        render_convenient,
        convert_rgb_to_saveable,
        convert_disparity_to_saveable,
        convert_disparity_to_jet,
        convert_disparity_to_phong,
        store_ray_bending_mesh_visualization,
        to8b,
    ) = _setup_nonrigid_nerf_network(args.input)
    print("sucessfully loaded nerf network", flush=True)

    # load dataset
    ray_bending_latents = (
        get_training_ray_bending_latents()
    )  # shape: frames x latent_size
    (
        images,
        poses,
        all_rotations,
        all_translations,
        bds,
        render_poses,
        render_rotations,
        render_translations,
        i_train,
        i_val,
        i_test,
        near,
        far,
        dataset_extras,
    ) = load_llff_dataset(
        render_kwargs_train_=render_kwargs_train, render_kwargs_test_=render_kwargs_test
    )  # load dataset that this nerf was trained on
    print("sucessfully loaded dataset", flush=True)

    # determine subset
    if args.deformations == "train":
        indices = i_train
        poses = poses[i_train]
        ray_bending_latents = ray_bending_latents[i_train]
        images = images[i_train]
        print("rendering training set")
    elif args.deformations == "test":
        indices = i_test
        poses = poses[i_test]
        ray_bending_latents = ray_bending_latents[i_test]
        images = images[i_test]
        print("rendering test set")
    elif args.deformations == "all":
        print("rendering training and test set")
    else:
        raise RuntimeError("invalid --deformations argument")

    copy_over_groundtruth_images = False
    if copy_over_groundtruth_images:
        groundtruth_images_folder = os.path.join(output_folder, "groundtruth")
        create_folder(groundtruth_images_folder)
        for i, rgb in enumerate(images):
            rgb = convert_rgb_to_saveable(rgb)
            file_prefix = os.path.join(groundtruth_images_folder, str(i).zfill(6))
            imageio.imwrite(file_prefix + ".png", rgb)

    # determine camera poses and latent codes
    num_poses = poses.shape[0]
    if args.camera_path == "input_recontruction":
        poses = poses
    elif args.camera_path == "fixed":
        poses = torch.stack(
            [torch.Tensor(poses[args.fixed_view]) for _ in range(num_poses)], 0
        )  # N x 3 x 4
    elif args.camera_path == "spiral":
        # poses = np.stack(_spiral_poses(poses, bds, num_poses), axis=0)
        poses = []
        while len(poses) < num_poses:
            poses += [render_pose for render_pose in render_poses]
        poses = np.stack(poses, axis=0)[:num_poses]
    else:
        # poses has shape N x ... and ray_bending_latents has shape N x ...
        # Can design custom camera paths here.
        raise RuntimeError
    latents = ray_bending_latents

    latents = latents.detach().cuda()

    # rendering
    correspondence_rgbs = []
    rigidities = []
    rgbs = []
    disps = []

    frames_at_a_time = 10
    num_output_frames = poses.shape[0]
    for start_index in range(0, num_output_frames, frames_at_a_time):

        end_index = np.min([start_index + frames_at_a_time, num_output_frames])
        print(
            "rendering "
            + str(start_index)
            + " to "
            + str(end_index)
            + " out of "
            + str(num_output_frames),
            flush=True,
        )

        subposes = poses[start_index:end_index]
        sublatents = [latents[i] for i in range(start_index, end_index)]

        # render
        subrgbs, subdisps, details_and_rest = render_convenient(
            poses=subposes,
            ray_bending_latents=sublatents,
            with_ray_bending=not args.render_canonical,
            detailed_output=True,
            rigidity_test_time_cutoff=args.forced_background_stabilization,
        )
        print("finished rendering", flush=True)

        rgbs += [image for image in subrgbs]
        disps += [image for image in subdisps]

        # determine correspondences
        # details_and_rest: list, one entry per image. each image has first two dimensions height x width.
        min_point = np.array(
            checkpoint_dict["scripts_dict"]["min_nerf_volume_point"]
        ).reshape(1, 1, 3)
        max_point = np.array(
            checkpoint_dict["scripts_dict"]["max_nerf_volume_point"]
        ).reshape(1, 1, 3)
        for i, image_details in enumerate(details_and_rest):
            # visibility_weight is the weight of the influence that each sample has on the final rgb value. so they sum to at most 1.
            accumulated_visibility = torch.cumsum(
                torch.Tensor(image_details["fine_visibility_weights"]).cuda(), dim=-1
            )  # height x width x point samples
            median_indices = torch.min(torch.abs(accumulated_visibility - 0.5), dim=-1)[
                1
            ]  # height x width. visibility goes from 0 to 1. 0.5 is the median, so treat it as "most likely to be on the actually visible surface"
            # visualize canonical correspondences as RGB
            height, width = median_indices.shape
            surface_pixels = (
                image_details["fine_input_pts"]
                .reshape(height * width, -1, 3)[
                    np.arange(height * width), median_indices.cpu().reshape(-1), :
                ]
                .reshape(height, width, 3)
            )  # height x width x 3. median_indices contains the index of one ray sample for each pixel. this ray sample is selected in this line of code.
            correspondence_rgb = (surface_pixels - min_point) / (max_point - min_point)
            number_of_small_rgb_voxels = 100  # break the canonical space into smaller voxels. each voxel covers the entire RGB space [0,1]^3. makes it easier to visualize small changes. leads to a 3D checkerboard pattern
            if number_of_small_rgb_voxels > 1:
                correspondence_rgb *= number_of_small_rgb_voxels
                correspondence_rgb = correspondence_rgb - correspondence_rgb.astype(int)
            correspondence_rgbs.append(correspondence_rgb)

            # visualize rigidity
            if "fine_rigidity_mask" in image_details:
                rigidity = (
                    image_details["fine_rigidity_mask"]
                    .reshape(height * width, -1)[
                        np.arange(height * width), median_indices.cpu().reshape(-1)
                    ]
                    .reshape(height, width)
                )  # height x width. values in [0,1]
                rigidities.append(rigidity)
            else:
                rigidities.append(None)

    rgbs = np.stack(rgbs, axis=0)
    disps = np.stack(disps, axis=0)
    correspondence_rgbs = np.stack(correspondence_rgbs, axis=0)
    use_rigidity = rigidities[0] is not None

    # store results
    # for i, (rgb, disp, correspondence_rgb, rigidity) in zip(indices, (zip(rgbs, disps, correspondence_rgbs, rigidities))):
    for i, (rgb, disp, correspondence_rgb, rigidity) in enumerate(
        zip(rgbs, disps, correspondence_rgbs, rigidities)
    ):
        print("storing image " + str(i) + " / " + str(rgbs.shape[0]), flush=True)
        rgb = convert_rgb_to_saveable(rgb)
        disp_saveable = convert_disparity_to_saveable(disp)
        disp_jet = convert_disparity_to_jet(disp)
        disp_phong = convert_disparity_to_phong(disp)
        correspondence_rgb = convert_rgb_to_saveable(correspondence_rgb)
        if use_rigidity:
            rigidity_saveable = convert_disparity_to_saveable(rigidity, normalize=False)
            rigidity_jet = convert_disparity_to_jet(rigidity, normalize=False)

        file_postfix = "_" + str(i).zfill(6) + ".png"
        imageio.imwrite(os.path.join(output_folder, "rgb" + file_postfix), rgb)
        imageio.imwrite(
            os.path.join(output_folder, "correspondences" + file_postfix),
            correspondence_rgb,
        )
        if use_rigidity:
            imageio.imwrite(
                os.path.join(output_folder, "rigidity" + file_postfix),
                rigidity_saveable,
            )
            imageio.imwrite(
                os.path.join(output_folder, "rigidity_jet" + file_postfix), rigidity_jet
            )
        imageio.imwrite(
            os.path.join(output_folder, "disp" + file_postfix), disp_saveable
        )
        imageio.imwrite(
            os.path.join(output_folder, "disp_jet" + file_postfix), disp_jet
        )
        imageio.imwrite(
            os.path.join(output_folder, "disp_phong" + file_postfix), disp_phong
        )

    # movies
    file_prefix = os.path.join(output_folder, "video_")
    try:
        print("storing RGB video...", flush=True)
        imageio.mimwrite(
            file_prefix + "rgb.mp4",
            convert_rgb_to_saveable(rgbs),
            fps=args.output_video_fps,
            quality=9,
        )
        print("storing correspondence RGB video...", flush=True)
        imageio.mimwrite(
            file_prefix + "correspondences.mp4",
            convert_rgb_to_saveable(correspondence_rgbs),
            fps=args.output_video_fps,
            quality=9,
        )
        print("storing disparity video...", flush=True)
        imageio.mimwrite(
            file_prefix + "disp.mp4",
            convert_disparity_to_saveable(disps),
            fps=args.output_video_fps,
            quality=9,
        )
        print("storing disparity jet video...", flush=True)
        imageio.mimwrite(
            file_prefix + "disp_jet.mp4",
            np.stack([convert_disparity_to_jet(disp) for disp in disps], axis=0),
            fps=args.output_video_fps,
            quality=9,
        )
        print("storing disparity phong video...", flush=True)
        imageio.mimwrite(
            file_prefix + "disp_phong.mp4",
            np.stack([convert_disparity_to_phong(disp) for disp in disps], axis=0),
            fps=args.output_video_fps,
            quality=9,
        )
        if use_rigidity:
            rigidities = np.stack(rigidities, axis=0)
            print("storing rigidity video...", flush=True)
            imageio.mimwrite(
                file_prefix + "rigidity.mp4",
                convert_disparity_to_saveable(rigidities, normalize=False),
                fps=args.output_video_fps,
                quality=9,
            )
            print("storing rigidity jet video...", flush=True)
            imageio.mimwrite(
                file_prefix + "rigidity_jet.mp4",
                np.stack(
                    [
                        convert_disparity_to_jet(rigidity, normalize=False)
                        for rigidity in rigidities
                    ],
                    axis=0,
                ),
                fps=args.output_video_fps,
                quality=9,
            )
    except:
        print("imageio.mimwrite() failed. maybe ffmpeg is not installed properly?")

    # error maps
    if args.camera_path == "input_recontruction":
        errors = []
        for i, (gt_image, rgb) in enumerate(zip(images, rgbs)):
            error = np.linalg.norm(gt_image - rgb, axis=-1) / np.sqrt(
                1 + 1 + 1
            )  # height x width
            error *= 10.0  # exaggarate error
            error = np.clip(error, 0.0, 1.0)
            error = convert_disparity_to_jet(
                error, normalize=False
            )  # height x width x 3. int values in [0,255]
            errors.append(error)
            imageio.imwrite(
                os.path.join(output_folder, "error_" + str(i).zfill(6) + ".png"), error
            )
        try:
            imageio.mimwrite(
                file_prefix + "error.mp4",
                np.stack(errors, axis=0),
                fps=args.output_video_fps,
                quality=9,
            )
        except:
            print("imageio.mimwrite() failed. maybe ffmpeg is not installed properly?")


if __name__ == "__main__":

    import configargparse

    parser = configargparse.ArgumentParser()
    # mandatory arguments
    parser.add_argument(
        "--input",
        type=str,
        help="the experiment folder that was created by train.py when training the network.",
    )
    parser.add_argument(
        "--deformations",
        type=str,
        help='"train", "test", "all". which deformations/time steps to render.',
    )
    parser.add_argument(
        "--camera_path",
        type=str,
        help='"input_recontruction", "fixed". camera path to use for re-rendering. optionally, implement "spiral", see README.md',
    )
    # optional camera path arguments
    parser.add_argument(
        "--fixed_view",
        type=int,
        default=0,
        help='only used for "fixed" camera_path. groundtruth camera view index that is used for fixed-view re-rendering. default is 0.',
    )
    # optional modifications
    parser.add_argument(
        "--forced_background_stabilization",
        type=float,
        default=None,
        help="prevents deformations of points that are more rigid than the provided threshold. needs to be manually determined. can be None or a float in [0,1]. default is None.",
    )
    parser.add_argument(
        "--render_canonical",
        action="store_true",
        help="render the canonical NeRF model, without ray bending.",
    )
    parser.add_argument(
        "--output_video_fps",
        type=int,
        default=5,
        help="frame rate of the generated output video. default is 5.",
    )

    args = parser.parse_args()

    free_viewpoint_rendering(args)
