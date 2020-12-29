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


def lens_distortion_calibration(args):
    import cv2

    # alpha = 1 # 0 crops a very tight image to eliminate as much invalid/black area as possible, 1 keeps all pixels from the input, which leads to a large invalid/black area. values in between 0 and 1 are possible.

    input_folder = os.path.join(args.input, "images")
    output_folder = args.output

    images = sorted(os.listdir(input_folder))
    images = [filename for filename in images if filename[-4:] in [".png", ".jpg"]]

    if args.visualize_detections:
        detected_folder = os.path.join(output_folder, "detected/")
        create_folder(detected_folder)

    # based on https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((args.checkerboard_width * args.checkerboard_height, 3), np.float32)
    objp[:, :2] = np.mgrid[
        0 : args.checkerboard_height, 0 : args.checkerboard_width
    ].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for i, filename in enumerate(images):
        print(str(i) + " / " + str(len(images)) + " " + filename, flush=True)
        img = cv2.imread(os.path.join(input_folder, filename))
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        corners = np.array([])
        ret, corners = cv2.findChessboardCorners(
            gray, (args.checkerboard_height, args.checkerboard_width), corners, flags=0
        )
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria
            )  # corners2 == corners
            imgpoints.append(corners)
            if args.visualize_detections:
                # Draw and display the corners
                cv2.drawChessboardCorners(
                    img,
                    (args.checkerboard_height, args.checkerboard_width),
                    corners2,
                    ret,
                )
                cv2.imwrite(os.path.join(detected_folder, filename), img)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)
    if args.visualize_detections:
        # cv2.destroyAllWindows() # uncomment if using imshow()
        pass

    print("computing calibration...", flush=True)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print("RMSE (in pixel units): " + str(ret), flush=True)
    newcameramtx = mtx
    roi = (0, 0, width, height)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), alpha, (width, height))

    calibration_paramters = {
        "newcameramtx": newcameramtx.tolist(),
        "mtx": mtx.tolist(),
        "dist": dist.tolist(),
        "roi": roi,
    }

    print("storing calibration...", flush=True)

    import json

    with open(os.path.join(output_folder, "lens_distortion.json"), "w") as json_file:
        json.dump(calibration_paramters, json_file, indent=4)

    if args.undistort_calibration_images:
        undistorted_folder = os.path.join(output_folder, "undistorted/")
        create_folder(undistorted_folder)
        for i, filename in enumerate(images):

            distorted_image = cv2.imread(os.path.join(input_folder, filename))
            newcameramtx = np.array(calibration_paramters["newcameramtx"])
            mtx = np.array(calibration_paramters["mtx"])
            dist = np.array(calibration_paramters["dist"])
            roi = np.array(calibration_paramters["roi"])

            # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

            # undistort
            undistorted_image = cv2.undistort(
                distorted_image, mtx, dist, None, newcameramtx
            )
            # crop the image
            x, y, w, h = roi
            undistorted_image = undistorted_image[y : y + h, x : x + w]
            output_file = os.path.join(undistorted_folder, filename)
            cv2.imwrite(output_file, undistorted_image)

            # rgb mask
            if i == 0:
                Image.fromarray(
                    (255.0 * (np.sum(undistorted_image == 0.0, axis=-1) != 3)).astype(
                        "uint8"
                    ),
                    mode="L",
                ).save(os.path.join(undistorted_folder, "mask.png"))


def video_preprocessing(args):

    video_path = args.input
    output_folder = args.output
    ffmpeg_path = args.ffmpeg_path

    # extract frames
    images_folder = os.path.join(output_folder, "images/")
    create_folder(images_folder)
    from subprocess import run, check_output, STDOUT, DEVNULL

    command = ""
    # command += "-i " + video_path + " -f image2 -qscale:v 1 -qmin 1 " + images_folder + "image%05d.jpg" # highest quality and all images.
    command += (
        "-i "
        + video_path
        + " -f image2 -qscale:v 2 -vf fps="
        + str(args.fps)
        + " "
        + images_folder
        + "image%05d.png"
    )
    # command += "-i " + video_path + ' -f image2 -qscale:v 2 -vf "fps=' + str(args.fps) + ', crop=in_w:3/4*in_h:0:in_h/4" ' + images_folder + "image%05d.png" # crop
    print(command, flush=True)
    try:
        ffmpeg_output = check_output([ffmpeg_path] + command.split(" "), stderr=STDOUT)
    except:
        run(ffmpeg_path + " " + command)

    # take care of failed frames
    failed_frames_folder = os.path.join(output_folder, "images_failed/")
    if os.path.exists(failed_frames_folder):
        failed_frame_names = os.listdir(failed_frames_folder)
        print(
            "detected failed frames, will delete: " + str(failed_frame_names),
            flush=True,
        )
        [
            os.remove(os.path.join(images_folder, failed_frame))
            for failed_frame in failed_frame_names
        ]

    # create videos using ffmpeg
    print("creating full-resolution RGB video...", flush=True)
    command = ""
    command += (
        "-framerate " + str(args.fps) + " -i " + images_folder + "image%05d.png -y "
    )  # -y overwrites existing files automatically
    command += os.path.join(output_folder, "rgb_scene_fullres.mp4")
    try:
        ffmpeg_output = check_output([ffmpeg_path] + command.split(" "), stderr=STDOUT)
    except:
        run(ffmpeg_path + " " + command)

    # print("creating downsampled RGB video...", flush=True)
    # command = ""
    # command += "-i " + os.path.join(output_folder, "rgb_scene_fullres.mp4") + ' -vf scale="iw/1:ih/2"' + " -y "
    # command += os.path.join(output_folder, "rgb_scene_downsampled.mp4")
    # ffmpeg_output = check_output([ffmpeg_path] + command.split(" "), stderr=STDOUT)


def _undistort_image(args):
    import cv2

    (
        i,
        distorted_images,
        undistorted_folder,
        distorted_folder,
        undistortion_parameters,
        mask_folder,
    ) = args

    filename = distorted_images[i]
    input_file = os.path.join(distorted_folder, filename)
    output_file = os.path.join(undistorted_folder, filename)
    print(" " + str(i) + "/" + str(len(distorted_images)), flush=True, end="")

    # get inputs
    distorted_image = cv2.imread(input_file)
    newcameramtx = np.array(undistortion_parameters["newcameramtx"])
    mtx = np.array(undistortion_parameters["mtx"])
    dist = np.array(undistortion_parameters["dist"])
    roi = np.array(undistortion_parameters["roi"])

    # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

    # undistort
    undistorted_image = cv2.undistort(distorted_image, mtx, dist, None, newcameramtx)
    # faster alternative:
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5) # do once before parallelization
    # dst = cv2.remap(img, mapx, mapy, cv.INTER_LINEAR) # do for each image in parallel
    # crop the image
    x, y, w, h = roi
    undistorted_image = undistorted_image[y : y + h, x : x + w]
    cv2.imwrite(output_file, undistorted_image)

    # rgb mask
    if i == 0:
        Image.fromarray(
            (255.0 * (np.sum(undistorted_image == 0.0, axis=-1) != 3)).astype("uint8"),
            mode="L",
        ).save(os.path.join(mask_folder, "mask.png"))


def undistort(args):

    input_folder = args.input
    output_folder = args.output
    undistortion_file = args.undistort_with_calibration_file
    if os.path.isdir(undistortion_file):
        undistortion_file = os.path.join(undistortion_file, "lens_distortion.json")

    import json

    with open(undistortion_file, "r") as undistortion_file:
        undistortion_parameters = json.load(undistortion_file)

    if os.path.normpath(input_folder) == os.path.normpath(output_folder):
        distorted_folder = os.path.join(output_folder, "distorted_images/")
        undistorted_folder = os.path.join(input_folder, "images/")
        # backup distorted images
        shutil.move(undistorted_folder, distorted_folder)
    else:
        distorted_folder = os.path.join(input_folder, "images/")
        undistorted_folder = os.path.join(output_folder, "images/")

    create_folder(undistorted_folder)

    mask_folder = undistorted_folder[:-1] + "_mask/"
    create_folder(mask_folder)

    # undistort images
    local_parallel_processes = 5
    distorted_images = [
        file for file in os.listdir(distorted_folder) if file[-4:] in [".png", ".jpg"]
    ]
    from multiprocessing import Pool

    with Pool(local_parallel_processes) as pool:
        pool.map(
            _undistort_image,
            [
                (
                    i,
                    distorted_images,
                    undistorted_folder,
                    distorted_folder,
                    undistortion_parameters,
                    mask_folder,
                )
                for i in range(len(distorted_images))
            ],
        )

    # store new intrinsics in file
    with open(
        os.path.join(output_folder, "undistorted_calibration.txt"), "w"
    ) as output_calibration:
        # see for indexing into matrix: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
        output_calibration.write(
            "color fx " + str(undistortion_parameters["newcameramtx"][0][0]) + "\n"
        )
        output_calibration.write(
            "color fy " + str(undistortion_parameters["newcameramtx"][1][1]) + "\n"
        )
        output_calibration.write(
            "color cx " + str(undistortion_parameters["newcameramtx"][0][2]) + "\n"
        )
        output_calibration.write(
            "color cy " + str(undistortion_parameters["newcameramtx"][1][2]) + "\n"
        )


def create_folder(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


def preprocess(args):

    # output folder
    if args.output is None:
        if os.path.isfile(args.input):
            input_folder, input_file = os.path.split(args.input)
            input_name, input_extension = os.path.splitext(input_file)
            args.output = os.path.join(input_folder, input_name)
        else:
            args.output = args.input
    create_folder(args.output)

    # video extraction
    if os.path.isfile(args.input):
        video_preprocessing(args)
        args.input = args.output

    if args.calibrate_lens_distortion:
        lens_distortion_calibration(args)
    else:
        # undistort input images with previously computed lens distortion parameters
        if args.undistort_with_calibration_file is not None:
            undistort(args)

        # get camera poses by running colmap
        from llff_preprocessing import gen_poses

        gen_poses(args.input, args.colmap_matching)


if __name__ == "__main__":

    import configargparse

    parser = configargparse.ArgumentParser()
    # mandatory arguments
    parser.add_argument(
        "--input",
        type=str,
        help='input. can be a video file or folder that contains a subfolder named "images", which contains images. e.g. set to foo/bar if images are in foo/bar/images/image0.png',
    )
    # optional custom paths
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help='custom output folder. similar to --input, needs to be foo/bar such that subfolders like "images" can be created as foo/bar/images/',
    )
    parser.add_argument(
        "--colmap_matching",
        type=str,
        default="sequential_matcher",
        help='"sequential_matcher" (default. for temporally ordered input, e.g. video) or "exhaustive_matcher" (each image is matched with every other image).',
    )
    parser.add_argument(
        "--ffmpeg_path",
        type=str,
        default="ffmpeg",
        help="path to ffmpeg executable. only used for video input.",
    )
    # video input
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="when using video input, the frame rate at which images should be extracted from the video",
    )
    # apply computed lens distortion to undistort the input
    parser.add_argument(
        "--undistort_with_calibration_file",
        type=str,
        default=None,
        help="path to lens_distortion.json, the lens distortion calibration file (computed with calibrate_lens_distortion) that will be used to undistort the input images before running colmap.",
    )
    # compute lens distortion parameters from a checkerboard sequence (mandatory arguments)
    parser.add_argument(
        "--calibrate_lens_distortion",
        action="store_true",
        help="computes lens distortion parameters to later undistort images. input sequence needs to contain a checkerboard that follows the OpenCV design. does not compute camera poses.",
    )
    parser.add_argument(
        "--checkerboard_width",
        type=int,
        default=5,
        help="checkerboard width for lens distortion calibration. number of squares.",
    )
    parser.add_argument(
        "--checkerboard_height",
        type=int,
        default=5,
        help="checkerboard height for lens distortion calibration. number of squares.",
    )
    # compute lens distortion parameters from a checkerboard sequence (optional arguments)
    parser.add_argument(
        "--visualize_detections",
        action="store_true",
        help="when calibrating lens distortion, output the checkerboard detection",
    )
    parser.add_argument(
        "--undistort_calibration_images",
        action="store_true",
        help="when calibrating lens distortion, undistort the calibration sequence afterwards to inspect how well undistortion works",
    )
    args = parser.parse_args()

    preprocess(args)
