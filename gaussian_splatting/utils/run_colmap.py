from argparse import ArgumentParser

import shutil
import urllib.request
import zipfile
from pathlib import Path

import enlighten

import pycolmap
from pycolmap import logging

def incremental_mapping_with_pbar(database_path, image_path, sfm_path):
    """
    Performs incremental Structure from Motion (SfM) on images using the pycolmap interface.#+
    Displays a progress bar to track the number of registered images.#+

    Parameters:
    - database_path (str or Path): Path to the database file where image metadata and matches are stored.#+
    - image_path (str or Path): Path to the directory containing the input images.#+

    Returns:
    - reconstructions (dict): A dictionary containing the reconstructed 3D models. Each key is the reconstruction ID,#+
      and the value is an instance of the pycolmap.Reconstruction class.#+
    """
    num_images = pycolmap.Database(database_path).num_images
    with enlighten.Manager() as manager:
        with manager.counter(
            total=num_images, desc="Images registered:"
        ) as pbar:
            pbar.update(0, force=True)
            reconstructions = pycolmap.incremental_mapping(
                database_path,
                image_path,
                sfm_path,
                initial_image_pair_callback=lambda: pbar.update(2),
                next_image_callback=lambda: pbar.update(1),
            )
            
    return reconstructions


def run(args):
    """
    This function orchestrates the entire Structure from Motion (SfM) and Multi-View Stereo (MVS) pipeline.
    It performs feature extraction, matching, incremental mapping, undistortion of images, patch-match stereo,
    and stereo fusion to generate a dense 3D reconstruction.

    Parameters:
    - args (argparse.Namespace): An object containing command-line arguments. It should have the following attributes:
      - output_path (str): The path to the output directory where all intermediate and final results will be stored.
      - image_path (str): The path to the directory containing the input images.

    Returns:
    None
    """
    output_path = Path(args.output_path)
    image_path = Path(args.image_path)
    if not image_path.is_dir():
        raise FileNotFoundError(f"The provided image path {image_path} does not exist or is not a directory.")

    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"
    mvs_path = output_path / "mvs"

    output_path.mkdir(exist_ok=True)
    # The log filename is postfixed with the execution timestamp.
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")

    if database_path.exists():
        database_path.unlink()
    pycolmap.set_random_seed(0)
    pycolmap.extract_features(database_path, image_path)
    pycolmap.match_exhaustive(database_path)

    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

    recs = incremental_mapping_with_pbar(database_path, image_path, sfm_path)
    # alternatively, use:
    # import custom_incremental_mapping
    # recs = custom_incremental_mapping.main(
    #     database_path, image_path, sfm_path
    # )
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")

    # dense reconstruction
    pycolmap.undistort_images(mvs_path, sfm_path / '0', image_path)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    options = pycolmap.StereoFusionOptions()
    options.num_threads = 6
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path,options=options)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser.add_argument(
        "--image-folder", required=True, type=str, help="path to image folder"
    )
    parser.add_argument("--output-path",required=True, type=str, help="output directory")
    
    run(parser.parse_args())
