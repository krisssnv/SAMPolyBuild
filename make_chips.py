"""Cut large georeferenced orthophotos into fixed-size 8-bit RGB chips for infer_auto.py.

Each chip is saved as <tile>__<x>_<y>.png, where x, y are the pixel offsets of the chip's
top-left corner in the source tile. chips_index.json records each tile's CRS and geotransform
so predicted polygons can be mapped back to map coordinates later.

Example:
    python make_chips.py --src "C:/path/to/tifs/A2-009-2021AA.tif" --out chips/
    python make_chips.py --src "C:/path/to/tifs" --out chips/ --size 512 --overlap 64
"""
import argparse
import glob
import json
import os
import os.path as osp

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Cut orthophotos into chips')
    parser.add_argument('--src', required=True, help='a .tif file or a folder of .tif files')
    parser.add_argument('--out', required=True, help='output folder for chips')
    parser.add_argument('--size', default=512, type=int, help='chip size in pixels')
    parser.add_argument('--overlap', default=64, type=int, help='overlap between neighbouring chips in pixels')
    parser.add_argument('--bands', default=[1, 2, 3], type=int, nargs=3, help='source bands used as R G B (1-based)')
    parser.add_argument('--pmin', default=1.0, type=float, help='lower percentile for the 8-bit stretch')
    parser.add_argument('--pmax', default=99.0, type=float, help='upper percentile for the 8-bit stretch')
    parser.add_argument('--range', default=None, type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='fixed stretch range for all tiles instead of per-tile percentiles, e.g. 7000 50000')
    parser.add_argument('--max_nodata', default=0.5, type=float,
                        help='skip chips where more than this fraction of pixels is nodata (0 in all bands)')
    return parser.parse_args()


def chip_offsets(length, size, overlap):
    """Start offsets covering [0, length); the last chip is shifted back to stay inside the image."""
    if length <= size:
        return [0]
    step = size - overlap
    offsets = list(range(0, length - size, step))
    offsets.append(length - size)
    return offsets


def stretch_limits(src, bands, pmin, pmax):
    """Per-band percentiles from a ~1000 px decimated read of the whole tile."""
    scale = max(src.width, src.height) / 1000
    shape = (len(bands), max(1, int(src.height / scale)), max(1, int(src.width / scale)))
    preview = src.read(bands, out_shape=shape).astype(np.float32)
    valid = preview.any(axis=0)
    lo = np.array([np.percentile(b[valid], pmin) for b in preview], dtype=np.float32)
    hi = np.array([np.percentile(b[valid], pmax) for b in preview], dtype=np.float32)
    return lo, hi


def to_uint8(data, lo, hi):
    data = (data.astype(np.float32) - lo[:, None, None]) / np.maximum(hi - lo, 1)[:, None, None]
    return (np.clip(data, 0, 1) * 255).round().astype(np.uint8)


def main():
    args = parse_args()
    if osp.isdir(args.src):
        tifs = sorted(glob.glob(osp.join(args.src, '*.tif')))
    else:
        tifs = [args.src]
    os.makedirs(args.out, exist_ok=True)
    index_pth = osp.join(args.out, 'chips_index.json')
    index = json.load(open(index_pth)) if osp.exists(index_pth) else {}

    for tif in tifs:
        name = osp.splitext(osp.basename(tif))[0]
        with rasterio.open(tif) as src:
            if args.range is not None:
                lo = np.full(3, args.range[0], dtype=np.float32)
                hi = np.full(3, args.range[1], dtype=np.float32)
            else:
                lo, hi = stretch_limits(src, args.bands, args.pmin, args.pmax)
            xs = chip_offsets(src.width, args.size, args.overlap)
            ys = chip_offsets(src.height, args.size, args.overlap)
            saved = 0
            for y in tqdm(ys, desc=name):
                for x in xs:
                    w = Window(x, y, min(args.size, src.width - x), min(args.size, src.height - y))
                    data = src.read(args.bands, window=w)
                    if 1 - data.any(axis=0).mean() > args.max_nodata:
                        continue
                    Image.fromarray(to_uint8(data, lo, hi).transpose(1, 2, 0)).save(
                        osp.join(args.out, f'{name}__{x}_{y}.png'))
                    saved += 1
            index[name] = dict(src=osp.abspath(tif), crs=src.crs.to_wkt(), transform=list(src.transform)[:6],
                               width=src.width, height=src.height, size=args.size, overlap=args.overlap,
                               stretch_lo=lo.tolist(), stretch_hi=hi.tolist())
        print(f'{name}: {saved} chips (of {len(xs) * len(ys)}), stretch {lo.round().tolist()} -> {hi.round().tolist()}')
        json.dump(index, open(index_pth, 'w'), indent=1)


if __name__ == '__main__':
    main()
