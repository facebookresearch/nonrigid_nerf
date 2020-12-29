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
