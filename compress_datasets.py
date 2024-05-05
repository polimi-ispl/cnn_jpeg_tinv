"""
A small utility script to compress the datasets

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries import
import pandas as pd
import os
import sys
import argparse
import glob
from tqdm import tqdm
from PIL import Image
import cv2
from isplutils.jpeg_compression import JPEGTransform
import ntpath
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from functools import partial


# --- Helpers & functions

def compress_image(item: Tuple[pd.Index, pd.Series], dest_dir: str, qf: int, csub: int=0, block_size: int=8,
                   backend:str = 'PIL', custom_transform: JPEGTransform = None) -> Tuple[pd.Index, pd.Series]:
    i, r = item
    try:
        # Load image
        img = Image.open(i[1])
        r['height'] = img.height
        r['width'] = img.width
        r['format'] = img.format
        r['mode'] = img.mode
        # Create save path
        dest_dir = os.path.join(dest_dir, i[0])
        os.makedirs(dest_dir, exist_ok=True)
        save_path = os.path.join(dest_dir, f'{ntpath.split(i[1])[1].split(".")[0]}.jpeg')
        if os.path.exists(save_path):
            r['jpeg_image_path'] = save_path
        else:
            if block_size != 8:
                save_path = save_path.replace('.jpeg', '.png')
                # Compress and save image with opencv at highest quality (we don't want other artifacts)
                img = custom_transform.transform(image=np.array(img))['image']
                cv2.imwrite(save_path, img)
                #Image.fromarray(img).save(save_path, subsampling=0, quality=100)
            else:
                if backend == 'PIL':
                    # Save image
                    img.save(save_path, subsampling=csub, quality=qf)
                elif backend == 'opencv':
                    # Save image
                    csub = {0: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444, 1: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
                            2: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420}[csub]
                    cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, qf, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, csub])
                elif backend == 'custom':
                    # Compress and save image with opencv at highest quality (we don't want other artifacts)
                    img = custom_transform.transform(image=np.array(img))['image']
                    cv2.imwrite(save_path.replace('.jpeg', '.png'), img)
                    #Image.fromarray(img).save(save_path, subsampling=0, quality=100)
            r['jpeg_image_path'] = save_path
    except Exception as e:
        print('Error while processing: {}'.format(i[1]))
        img = Image.open(i[1])
        img = custom_transform.transform(image=np.array(img))['image']
#         print("-" * 60)
#         traceback.print_exc(file=sys.stdout, limit=5)
#         print("-" * 60)
        r['jpeg_image_path'] = 'EMPTY'
    return [i, r]


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, help='Directory containing the uncompressed images divided in folds',
                        default='/nas/home/ecannas/jpeg_expl/data/uncompressed')
    parser.add_argument('--dest_dir', type=str, help='Directory where to store the compressed images divided in folds',
                        default='/nas/home/ecannas/jpeg_expl/data/compressed')
    parser.add_argument('--qf', type=int, help='Quality factor for compression', default=50)
    parser.add_argument('--csub', type=int, help='Whether to subsample the chroma-luminance subspace or not',
                        choices=[0, 1, 2], default=0)
    parser.add_argument('--num_threads', type=int, help='Number of threads to use to compress in //', default=20)
    parser.add_argument('--batch_size', type=int, help='Number of images to compress in // at a time', default=16)
    parser.add_argument('--backend', type=str, help='Which backend to use for the compression', default='PIL',
                        choices=['PIL', 'opencv', 'custom'])
    parser.add_argument('--block_size', type=int, help='Size of the block for the computation of the DCT',
                        default=8)

    return parser.parse_args(argv)


# --- Main
def main(argv):

    # Parse arguments
    args = parse_args(argv)
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    qf = args.qf
    csub = args.csub
    block_size = args.block_size
    backend = args.backend

    # Instantiate the custom JPEG transform to avoid waste of times in the instantiation
    ratio = {0: '4:4:4', 1: '4:2:2', 2: '4:2:0'}[csub]
    custom_transform = JPEGTransform(qf=qf, downsampling_strategy=ratio, block_size=block_size)

    # Let's create a DataFrame and add some info on the images in the source folder
    all_imgs = glob.glob(os.path.join(source_dir, '**', '*.*'), recursive=True)
    df = pd.DataFrame(index=pd.Index(all_imgs, name='path'))
    df['folder'] = [path.split('/')[-2] for path in df.index.tolist()]
    df = df.set_index('folder', append=True, drop=True).swaplevel(0, 1)

    # Let's compress the iamges in // and add some useful info
    df['height'] = 0
    df['width'] = 0
    df['format'] = ''
    df['jpeg_image_path'] = ''
    df['mode'] = ''
    num_threads = args.num_threads
    batch_size = args.batch_size
    dest_dir = os.path.join(dest_dir, backend, f'qf-{qf}_subsampling-{csub}_block_size-{block_size}')
    os.makedirs(dest_dir, exist_ok=True)
    # with ThreadPoolExecutor(num_threads) as p:
    #     for batch_idx0 in tqdm(np.arange(start=0, stop=len(df), step=batch_size), desc='Compression images'):
    #         to_save_rows = list(p.map(partial(compress_image, dest_dir=dest_dir, qf=qf, csub=csub, backend=backend,
    #                                           block_size=block_size, custom_transform=custom_transform),
    #                                   df.iloc[batch_idx0:batch_idx0+batch_size].iterrows()))
    #         for elem in to_save_rows:
    #             i, r = elem
    #             df.loc[i] = r

    for batch_idx0 in tqdm(np.arange(start=0, stop=len(df), step=batch_size), desc='Compression images'):
        to_save_rows = list(map(partial(compress_image, dest_dir=dest_dir, qf=qf, csub=csub, backend=backend,
                                          block_size=block_size, custom_transform=custom_transform),
                                        df.iloc[batch_idx0:batch_idx0+batch_size].iterrows()))
        for elem in to_save_rows:
            i, r = elem
            df.loc[i] = r

    # Save the results
    df.to_pickle(os.path.join(dest_dir, f'compression_df.pkl'))


if __name__=='__main__':
    main(sys.argv[1:])




