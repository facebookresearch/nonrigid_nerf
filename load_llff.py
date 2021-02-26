import numpy as np
import os


def load_llff_data(datadir, factor, recenter, bd_factor, spherify):

    import json

    with open("./data/example_sequence/precomputed.json", "r") as json_file:
        precomputed = json.load(json_file)
    poses = np.array(precomputed["poses"])
    bds = np.array(precomputed["bds"])
    render_poses = np.array(precomputed["render_poses"])
    i_test = precomputed["i_test"]

    import imageio

    images = sorted(os.listdir("./data/example_sequence/images/"))
    images = (
        np.stack(
            [
                imageio.imread(
                    os.path.join("./data/example_sequence/images/", image),
                    ignoregamma=True,
                )
                for image in images
            ],
            axis=-1,
        )
        / 255.0
    )
    images = np.moveaxis(images, -1, 0).astype(np.float32)

    return images, poses, bds, render_poses, i_test

def load_llff_data_multi_view(datadir, factor, recenter, bd_factor, spherify):
   
    import imageio
    images = sorted(os.listdir(os.path.join(datadir, "images")))
    images = (
        np.stack(
            [
                imageio.imread(
                    os.path.join(datadir, "images", image),
                    ignoregamma=True,
                )
                for image in images
            ],
            axis=-1,
        )
        / 255.0
    )
    images = np.moveaxis(images, -1, 0).astype(np.float32)

    from train import _get_multi_view_helper_mappings
    extras = _get_multi_view_helper_mappings(len(images), datadir)

    import json
    with open(os.path.join(datadir, "calibration.json"), "r") as json_file:
        calibration = json.load(json_file)
    poses = np.zeros((len(images), 3, 5))
    hwf = np.array([0., 0., 0.])
    for i in range(poses.shape[0]):
        raw_view = extras["raw_views"][extras["imageid_to_viewid"][i]]
        poses[i,:3,:3] = np.array(calibration[raw_view]["rotation"])
        poses[i,:3,3] = np.array(calibration[raw_view]["translation"])
        poses[i,:3,4] = hwf
    bds = np.array([calibration["min_bound"], calibration["max_bound"]])
    
    render_poses = poses.copy() # dummy
    i_test = 0 # dummy

    return images, poses, bds, render_poses, i_test
