import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from frames_eval import Evdet200kCocoDataset, letterbox_collate_fn
from loguru import logger

OUT_DIR = 'viz_inputs'
DATASET_ROOT_DIR = '/home/lhl/Git/datasets/EvDET200K'
BATCH_SIZE = 2
SEQ_LEN = 3


def save_image_array(arr, path):
    # arr: HxWxC, uint8
    img = Image.fromarray(arr)
    img.save(path)


def make_grid(img_paths, out_path):
    imgs = [Image.open(p) for p in img_paths]
    widths, heights = zip(*(i.size for i in imgs))
    total_w = sum(widths)
    max_h = max(heights)
    grid = Image.new('RGB', (total_w, max_h), (0,0,0))
    x = 0
    for im in imgs:
        grid.paste(im, (x,0))
        x += im.size[0]
    grid.save(out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cpu')

    logger.info('Loading dataset...')
    ds = Evdet200kCocoDataset(DATASET_ROOT_DIR, split='test', seq_len=SEQ_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=letterbox_collate_fn)

    logger.info('Fetching one batch and saving 640x640 inputs...')
    for batch_idx, (images_batch, targets, img_infos, ratios) in enumerate(dl):
        # images_batch shape: [B, SEQ_LEN, 3, H, W]
        B = images_batch.shape[0]
        S = images_batch.shape[1]
        H = images_batch.shape[3]
        W = images_batch.shape[4]

        logger.info(f'batch {batch_idx}: B={B}, S={S}, H={H}, W={W}, dtype={images_batch.dtype}')

        saved = []
        for b in range(B):
            sample_paths = []
            for s in range(S):
                arr = images_batch[b, s].numpy()  # C,H,W
                arr = np.transpose(arr, (1,2,0))  # H,W,C
                # clip/convert
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                fname = os.path.join(OUT_DIR, f'b{b:02d}_frame{s:02d}.png')
                save_image_array(arr, fname)
                sample_paths.append(fname)
                logger.info(f'Saved {fname} shape={arr.shape} min={arr.min()} max={arr.max()}')

            # make grid of seq frames for this sample
            grid_path = os.path.join(OUT_DIR, f'b{b:02d}_seq_grid.png')
            make_grid(sample_paths, grid_path)
            saved.append(grid_path)

        # make a multi-sample grid (concatenate vertically)
        grids = [Image.open(p) for p in saved]
        widths, heights = zip(*(g.size for g in grids))
        max_w = max(widths)
        total_h = sum(heights)
        vertical = Image.new('RGB', (max_w, total_h), (0,0,0))
        y = 0
        for g in grids:
            vertical.paste(g, (0, y))
            y += g.size[1]
        vertical.save(os.path.join(OUT_DIR, 'batch_seq_vertical.png'))

        logger.info(f'Wrote visualizations to {OUT_DIR}')
        break


if __name__ == '__main__':
    main()
