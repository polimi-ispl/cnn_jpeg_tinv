"""
File for the implementation of a custom JPEG compression.
Majority of the code is taken from https://github.com/katieshiqihe/image_compression

Authors:
Katie He - https://github.com/katieshiqihe
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Sara Mandelli - sara.mandelli@polimi.it

"""

# --- Libraries import
import numpy as np
from scipy.fft import dct, idct
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from isplutils.PatchExtractor import PatchExtractor
from typing import Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import cv2
from multiprocessing import cpu_count


# --- Helpers' functions and classes
class DownSampling():
    """
    Class for performing the Luma-Chrominance subsampling
    """
    def __init__(self, ratio='4:2:0'):
        """
        Constructor
        :param ratio: ratio for luma-chromincance channels subsampoing
        """
        assert ratio in ('4:4:4', '4:2:2', '4:2:0'), "Please choose one of the following {'4:4:4', '4:2:2', '4:2:0'}"
        self.ratio = ratio

    def __call__(self, x):
        """

        :param x: np.array, the channel considered for subsampling
        :return: np.array, channel subsampled
        """
        # No subsampling
        if self.ratio == '4:4:4':
            return x
        else:
            # Downsample with a window of 2 in the horizontal direction
            if self.ratio == '4:2:2':
                kernel = np.array([[0.5], [0.5]])
                out = np.repeat(convolve2d(x, kernel, mode='valid')[::2, :], 2, axis=0)
            # Downsample with a window of 2 in both directions
            else:
                kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
                out = np.repeat(np.repeat(convolve2d(x, kernel, mode='valid')[::2, ::2], 2, axis=0), 2, axis=1)
            return np.round(out).astype('int')


class DCT2D():
    """
    Wrapper class around the DCT transform
    """
    def __init__(self, norm='ortho'):
        """
        Constructor
        :param norm: define the norm for computing the DCT
        """
        if norm is not None:
            assert norm == 'ortho', "norm needs to be in {None, 'ortho'}"
        self.norm = norm

    def forward(self, x):
        """
        DCT transform
        :param x: np.array, the image block of whom we want to compute the coefficients
        :return: np.array, DCT coefficients
        """
        out = dct(dct(x, norm=self.norm, axis=0), norm=self.norm, axis=1)
        return out

    def backward(self, x):
        """
        Inverse DCT transform
        :param x: np.array, the DCT coefficients we want to transform back to image block
        :return: np.array, the rounded image block
        """
        out = idct(idct(x, norm=self.norm, axis=0), norm=self.norm, axis=1)
        return np.round(out)


class Quantization():
    """
    Wrapper class for the quantization of the blocks
    Quantization matrices taken from https://www.impulseadventure.com/photo/jpeg-quantization.html
    """

    def __init__(self, qf: int = 50, block_size: int = 8):
        """
        Constructor.
        IF the block size we are considering is != 8, we will compute an interpolating function on the
        quantization matrices for the new block size.
        :param qf: int, the quality factor to use in the quantization
        :param block_size: int, the size we are computing the JPEG on
        """
        # Luminance
        self.Q_lum = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                          [12, 12, 14, 19, 26, 58, 60, 55],
                          [14, 13, 16, 24, 40, 57, 69, 56],
                          [14, 17, 22, 29, 51, 87, 80, 62],
                          [18, 22, 37, 56, 68, 109, 103, 77],
                          [24, 35, 55, 64, 81, 104, 113, 92],
                          [49, 64, 78, 87, 103, 121, 120, 101],
                          [72, 92, 95, 98, 112, 100, 103, 99]])
        # Chrominance
        self.Q_chr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                          [18, 21, 26, 66, 99, 99, 99, 99],
                          [24, 26, 56, 99, 99, 99, 99, 99],
                          [47, 66, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99]])

        # Interpolate matrix in case block_size is not equal to 8x8
        if block_size != 8:
            X_new, Y_new = np.meshgrid(np.linspace(0, 7, block_size), np.linspace(0, 7, block_size))
            X, Y = np.mgrid[0:8:1, 0:8:1]
            positions = np.vstack([X.ravel(), Y.ravel()]).transpose()
            positions_x = positions[:, 1]
            positions_y = positions[:, 0]
            self.Q_lum = griddata((positions_y, positions_x), self.Q_lum[positions_y, positions_x], (Y_new, X_new),
                                  method='cubic')
            self.Q_chr = griddata((positions_y, positions_x), self.Q_chr[positions_y, positions_x], (Y_new, X_new),
                                  method='cubic')
            
        # Adjust matrices to specific quality factor
        # We are using the algorithm from the Independent JPEG Group (IJG, http://www.ijg.org/) using the intuition
        # reported here https://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression
        if qf < 50:
            S = 5000 / qf
        else:
            S = 200 - 2*qf
        self.Q_lum = np.floor((self.Q_lum * S + 50) / 100)
        self.Q_chr = np.floor((self.Q_chr * S + 50) / 100)
        self.Q_lum[self.Q_lum == 0] = 1  # avoid divide by zero
        self.Q_chr[self.Q_chr == 0] = 1

    def forward(self, x, channel_type):
        assert channel_type in ('lum', 'chr')

        if channel_type == 'lum':
            Q = self.Q_lum
        else:
            Q = self.Q_chr

        out = np.round(x / Q)
        return out

    def backward(self, x, channel_type):
        assert channel_type in ('lum', 'chr')

        if channel_type == 'lum':
            Q = self.Q_lum
        else:
            Q = self.Q_chr

        out = x * Q
        return out


class JPEGTransform():
    """
    Putting together all pieces above
    """
    def __init__(self, qf: int = 50, block_size: int = 8, downsampling_strategy: str = '4:2:0'):
        """
        Constructor.
        We store and initialize all the other members here (hopefully to reduce execution times)
        :param qf:
        :param block_size:
        :param sampling_ratio:
        """
        self.qf = qf
        self.block_size = block_size
        self.downsampling_strategy = downsampling_strategy
        self.dct_transform = DCT2D(norm='ortho')
        self.quantizer = Quantization(block_size=block_size, qf=qf)
        self.downsampler_444 = DownSampling(ratio='4:4:4')
        self.downsampler_420 = DownSampling(ratio='4:2:0')
        self.pe = PatchExtractor(dim=(self.block_size, self.block_size, 3), padding='reflect')

    def block_processing(self, channel_idx: int, block: np.array) -> np.array:
        """
        Function for the compression of a single JPEG block
        :param block_list: Tuple[int, np.array], tuple containing channel idx (for luma) and block for compression
        :return: np.array, block of encoded and decoded pixels using the JPEG transform
        """
        # DCT encoding
        encoded = self.dct_transform.forward(block)

        # Quantization
        channel_type = 'lum' if channel_idx == 0 else 'chr'
        encoded_quantized = self.quantizer.forward(encoded, channel_type)

        # De-quantization
        decoded = self.quantizer.backward(encoded_quantized, channel_type)

        # Reverse DCT
        return self.dct_transform.backward(decoded)

    def transform(self, image: np.array) -> Dict[str, np.array]:
        """
        Apply JPEG transform
        :param image:
        :param block_size:
        :param downsampling_strategy:
        :param qf: int, quality factor for the compression
        :return:
        """

        # --- Preprocessing

        # Colorspace transform (RGB -> YCrCb)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(int)

        # Mean scaling
        img -= 128

        # Luma-chrominance downsampling
        if self.downsampling_strategy == '4:4:4':
            img = np.stack((self.downsampler_444(img[:, :, channel]) for channel in range(img.shape[-1])), axis=2)
        else:
            # Fall back to 4:2:0 for all other options
            Y = self.downsampler_444(img[:, :, 0])
            Cr = self.downsampler_420(img[:, :, 1])
            Cb = self.downsampler_420(img[:, :, 2])
            img = np.stack((Y, Cr, Cb), axis=2)

        # Blocks extraction
        blocks = self.pe.extract(img)
        rec_shape = blocks.shape
        blocks = blocks.reshape((-1, self.block_size, self.block_size, 3))

        # --- Compression
        # Use Pool from the multiprocessing library because the compression task is
        # highly parallelizable. The same operation is performed on different blocks
        # where there is no dependency among the data.
        # with ThreadPoolExecutor(cpu_count()) as p:
        #     compressed_blocks = np.array(list(p.map(self.block_processing, blocks_list)))
        compressed_blocks = []
        for channel_idx in range(3):
            channel_blocks = []
            for block in blocks[:, :, :, channel_idx]:
                channel_blocks.append(self.block_processing(channel_idx, block)[np.newaxis, :, :, np.newaxis])
            compressed_blocks.append(np.concatenate(channel_blocks, axis=0))

        # --- Post-processing

        # Reconstruct image from blocks
        img = self.pe.reconstruct(np.concatenate(compressed_blocks, axis=-1).reshape(rec_shape))

        # Inverse mean scaling, cast as uint8 (clip first to avoid ugly rounding artifacts)
        img = np.clip(img+128, 0, 255).astype(np.uint8)

        # Back to RGB space
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB).astype(np.uint8)

        return {'image': img}

    def __call__(self, image: np.array, **kwargs) -> Dict[str, np.array]:
        return self.transform(image=image)


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from skimage.metrics import peak_signal_noise_ratio as PSNR
    from albumentations.augmentations.transforms import ImageCompression
    import time

    """
    Simple check to see if the compression is working fine
    """
    # Load DataFrame with images info
    df_path = '/nas/home/ecannas/jpeg_expl/data/compressed_opencv/qf-50_subsampling-0/compression_df_QF_50_subsampling_0.pkl'
    df = pd.read_pickle(df_path)

    # Load a random sample
    np.random.seed(42)  # setting the seed for sampling
    sample = df.sample(n=1)
    #uncompressed = cv2.cvtColor(cv2.imread(sample.index.get_level_values(1)[0]), cv2.COLOR_BGR2RGB)
    uncompressed = cv2.cvtColor(cv2.imread('/nas/home/ecannas/jpeg_expl/data/uncompressed/ucid/ucid00227.tif'), cv2.COLOR_BGR2RGB)

    # Let's try the JPEG compression from albumentations and ours
    qf = 50
    ratio = '4:2:0'
    for block_size in [6, 8, 10]:
        a_img_compression = ImageCompression(quality_lower=qf, quality_upper=qf,
                                             always_apply=True, p=1)
        our_img_compression = JPEGTransform(qf=qf, downsampling_strategy=ratio, block_size=block_size)
        tic = time.time()
        a_compressed = a_img_compression(image=uncompressed[0:256, 0:256])['image']
        print(f'{time.time()-tic} seconds for compression with Albumentations')
        tic = time.time()
        our_compressed = our_img_compression.transform(uncompressed[0:256, 0:256])['image']
        print(f'{time.time() - tic} seconds for compression with our custom function')

        # Let's compare the two images
        fig, axs = plt.subplots(3, 3, figsize=(36, 36))
        axs[0][0].imshow(uncompressed[:256, :256]), axs[0][0].set_title('Uncompressed image')
        axs[0][1].imshow(a_compressed), axs[0][1].set_title('Albumentations compression (OpenCV),\n'
                                                      f'PSNR with original = {PSNR(uncompressed[0:256, 0:256], a_compressed):.4f}')
        axs[0][2].imshow(our_compressed), axs[0][2].set_title(f'Custom compression,\n'
                                                        f'PSNR with original = {PSNR(uncompressed[0:256, 0:256], our_compressed):.4f}\n'
                                                        f'PSNR with Albumentations = {PSNR(a_compressed, our_compressed):.4f}')
        # Let's also have a closer look
        small_ps = 64
        axs[1][0].imshow(uncompressed[:small_ps, :small_ps]), axs[1][0].set_title('Uncompressed image')
        axs[1][1].imshow(a_compressed[:small_ps, :small_ps]), axs[1][1].set_title('Albumentations compression (OpenCV),\n'
                                                      f'PSNR with original = {PSNR(uncompressed[0:256, 0:256], a_compressed):.4f}')
        axs[1][2].imshow(our_compressed[:small_ps, :small_ps]), axs[1][2].set_title(f'Custom compression,\n'
                                                        f'PSNR with original = {PSNR(uncompressed[0:256, 0:256], our_compressed):.4f}\n'
                                                        f'PSNR with Albumentations = {PSNR(a_compressed, our_compressed):.4f}')
        # Even closer
        small_ps = 32
        axs[2][0].imshow(uncompressed[:small_ps, :small_ps]), axs[2][0].set_title('Uncompressed image')
        axs[2][1].imshow(a_compressed[:small_ps, :small_ps]), axs[2][1].set_title(
            'Albumentations compression (OpenCV),\n'
            f'PSNR with original = {PSNR(uncompressed[0:256, 0:256], a_compressed):.4f}')
        axs[2][2].imshow(our_compressed[:small_ps, :small_ps]), axs[2][2].set_title(f'Custom compression,\n'
                                                                                    f'PSNR with original = {PSNR(uncompressed[0:256, 0:256], our_compressed):.4f}\n'
                                                                                    f'PSNR with Albumentations = {PSNR(a_compressed, our_compressed):.4f}')
        plt.show()
