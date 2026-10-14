"""Predict building polygons on large georeferenced orthophotos without writing chips to disk.

Each tile is read window by window (see make_chips.py for the chipping/stretch options), the chips
are run through the auto-mode model in batches, and the chip polygons are shifted back into tile
pixel space, deduplicated across chip overlaps and written per tile as
    <out>/<tile>.shp         one record per building in the tile's CRS (id, score, score_cls, area_m2, tile);
                             <tile>.gpkg instead with --format gpkg
    <out>/<tile>_mask.tif    uint8 building mask (1 = building) on the tile's grid

The outputs get the tile's own CRS; tiles without one (only a .tfw) get --crs instead.

Example:
    python predict_tiles.py --src "C:/path/to/tifs/A2-009-2021AA.tif" --out predictions/
    python predict_tiles.py --src "C:/path/to/tifs" --out predictions/ --score_thr 0.3
    python predict_tiles.py --src a.tif b.tif "C:/other/tifs/D6-*.tif" --out predictions/
    python predict_tiles.py --run predict_tiles.yaml
In a --run YAML file the keys are the argument names below; arguments given on the command line override it.
"""
import argparse
import glob
import os
import os.path as osp
import time

import geopandas as gpd
import numpy as np
import rasterio
import torch
import yaml
from mmengine.config import Config, ConfigDict
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely import STRtree, affinity
from shapely.geometry import Polygon
from tqdm import tqdm

from make_chips import chip_offsets, stretch_limits, to_uint8
from mmdet.datasets.transforms import PackDetInputs, Resize
from mmpl.evaluation.metrics.coco_pl_metric import iou_scores
from mmpl.models.builder import build_pler
from mmpl.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Predict building polygons on large orthophotos')
    parser.add_argument('--run', default=None, help='YAML file with any of the arguments below')
    parser.add_argument('--src', nargs='+', help='.tif files, folders of .tif files and/or glob patterns')
    parser.add_argument('--out', help='output folder')
    parser.add_argument('--crs', default=None, help='CRS for tiles that have none of their own, e.g. EPSG:32635')
    parser.add_argument('--config', default='configs/auto_whumix.py')
    parser.add_argument('--ckpt_path', default='auto_whumix.pth')
    parser.add_argument('--size', default=512, type=int, help='chip size in pixels')
    parser.add_argument('--overlap', default=64, type=int,
                        help='chip overlap in pixels; buildings larger than this may come out split')
    parser.add_argument('--bands', default=[1, 2, 3], type=int, nargs=3, help='source bands used as R G B (1-based)')
    parser.add_argument('--pmin', default=1.0, type=float, help='lower percentile for the 8-bit stretch')
    parser.add_argument('--pmax', default=99.0, type=float, help='upper percentile for the 8-bit stretch')
    parser.add_argument('--range', default=None, type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='fixed stretch range for all tiles instead of per-tile percentiles')
    parser.add_argument('--max_nodata', default=0.5, type=float, help='skip chips with more nodata than this fraction')
    parser.add_argument('--score_thr', default=0.1, type=float, help='minimum classification score (score_cls)')
    parser.add_argument('--min_area', default=4.0, type=float, help='drop polygons smaller than this, in m2')
    parser.add_argument('--dup_iou', default=0.3, type=float,
                        help='polygons from overlapping chips covering more than this fraction of each other are merged')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--format', default='shp', choices=['shp', 'gpkg'], help='polygon output format')
    parser.add_argument('--skip_existing', action='store_true', help='skip tiles whose polygon output already exists')
    args = parser.parse_args()
    if args.run is not None:
        with open(args.run) as f:
            run = yaml.safe_load(f) or {}
        unknown = set(run) - {a.dest for a in parser._actions} - {'run'}
        if unknown:
            parser.error(f'unknown keys in {args.run}: {", ".join(sorted(unknown))}')
        if isinstance(run.get('src'), str):
            run['src'] = [run['src']]
        parser.set_defaults(**run)
        args = parser.parse_args()
    if not args.src or not args.out:
        parser.error('--src and --out are required (on the command line or in the --run file)')
    return args


def find_tifs(src):
    tifs = []
    for s in src:
        if osp.isdir(s):
            tifs += sorted(glob.glob(osp.join(s, '*.tif')))
        elif any(c in s for c in '*?['):
            tifs += sorted(glob.glob(s))
        elif osp.isfile(s):
            tifs.append(s)
        else:
            raise FileNotFoundError(s)
    return list(dict.fromkeys(osp.normpath(t) for t in tifs))  # drop duplicates, keep order


def build_model(config, ckpt_path, device):
    cfg = Config.fromfile(config)
    model_cfg = cfg.model_cfg
    model_cfg.hyperparameters.evaluator = None
    model_cfg.backbone.checkpoint = None
    model_cfg.panoptic_head.test_cfg.rcnn.max_per_img = 100
    model_cfg.panoptic_head.roi_head.multi_process = True
    model_cfg.update({'config_cfg': ConfigDict(cfg.to_dict())})
    model = build_pler(model_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    return model.to(device).eval()


class ChipPacker:
    """Same preprocessing as infer_auto.py's predict pipeline, minus reading the image from file."""

    def __init__(self):
        self.resize = Resize(scale=(1024, 1024))
        self.pack = PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))

    def __call__(self, rgb, img_id):
        img = np.ascontiguousarray(rgb[..., ::-1])  # the data preprocessor expects BGR like cv2-loaded images
        results = dict(img=img, img_id=img_id, img_path=img_id, img_shape=img.shape[:2], ori_shape=img.shape[:2])
        return self.pack(self.resize(results))


@torch.no_grad()
def predict_batch(model, packed, score_thr):
    """Returns per chip a list of (polygon ndarray (n, 2) in chip pixels, score, score_cls)."""
    batch = dict(inputs=[p['inputs'] for p in packed], data_samples=[p['data_samples'] for p in packed])
    data = model.data_preprocessor(batch, False)
    feats = model.extract_feat(data['inputs'])
    results = model.panoptic_head.predict(feats, data['data_samples'], model.backbone)
    out = []
    for res in results:
        pred = res.pred_instances
        chip = []
        if len(pred) > 0:
            masks = pred['masks'].cpu().numpy()
            scores_cls = pred['scores'].cpu().numpy()
            # combined score as in CocoPLMetric: classification, mask and polygon-vs-mask IoU
            scores = np.mean([scores_cls, pred['mask_scores'], iou_scores(pred['polygons'], masks)], axis=0)
            for poly, score, score_cls in zip(pred['polygons'], scores, scores_cls):
                if poly is not None and score_cls >= score_thr and len(poly) >= 3:
                    chip.append((poly, float(score), float(score_cls)))
        out.append(chip)
    return out


def touches_inner_edge(poly, x, y, size, width, height, tol=2):
    """Whether the polygon touches a chip edge that is not also a tile edge, i.e. it may be cut off."""
    minx, miny, maxx, maxy = poly.bounds
    return ((minx <= x + tol and x > 0) or (miny <= y + tol and y > 0) or
            (maxx >= x + size - tol and x + size < width) or (maxy >= y + size - tol and y + size < height))


def merge_overlaps(items, dup_iou):
    """Greedy dedup across chip overlaps. Complete (not edge-cut) and higher-scoring polygons win;
    two edge-cut pieces of the same building are unioned."""
    items = sorted(items, key=lambda it: (it['cut'], -it['score']))
    if not items:
        return []
    tree = STRtree([it['geometry'] for it in items])
    kept = set()
    for i, it in enumerate(items):
        geom = it['geometry']
        hit = None
        for j in sorted(tree.query(geom)):
            if j not in kept:
                continue
            other = items[j]['geometry']
            if geom.intersection(other).area / max(min(geom.area, other.area), 1e-6) > dup_iou:
                hit = j
                break
        if hit is None:
            kept.add(i)
        elif it['cut'] and items[hit]['cut']:
            union = items[hit]['geometry'].union(geom)
            if union.geom_type == 'Polygon':
                items[hit]['geometry'] = union
    return [items[i] for i in sorted(kept)]


def predict_tile(tif, args, model, packer, device):
    name = osp.splitext(osp.basename(tif))[0]
    with rasterio.open(tif) as src:
        crs = src.crs
        if crs is None:
            if args.crs is None:
                print(f'{name}: WARNING no CRS in the tif and no --crs given, outputs will have no CRS')
            else:
                crs = CRS.from_user_input(args.crs)
                print(f'{name}: no CRS in the tif, using {args.crs}')
        if args.range is not None:
            lo = np.full(3, args.range[0], dtype=np.float32)
            hi = np.full(3, args.range[1], dtype=np.float32)
        else:
            lo, hi = stretch_limits(src, args.bands, args.pmin, args.pmax)
        windows = [(x, y) for y in chip_offsets(src.height, args.size, args.overlap)
                   for x in chip_offsets(src.width, args.size, args.overlap)]
        items = []
        pending = []

        def flush():
            for (x, y), chip in zip([p[0] for p in pending], predict_batch(model, [p[1] for p in pending], args.score_thr)):
                for poly, score, score_cls in chip:
                    geom = Polygon(poly).buffer(0)
                    if geom.is_empty or geom.geom_type != 'Polygon':
                        continue
                    geom = affinity.translate(geom, x, y)
                    items.append(dict(geometry=geom, score=score, score_cls=score_cls,
                                      cut=touches_inner_edge(geom, x, y, args.size, src.width, src.height)))
            pending.clear()

        for x, y in tqdm(windows, desc=name):
            data = src.read(args.bands, window=Window(x, y, min(args.size, src.width - x), min(args.size, src.height - y)))
            if 1 - data.any(axis=0).mean() > args.max_nodata:
                continue
            pending.append(((x, y), packer(to_uint8(data, lo, hi).transpose(1, 2, 0), f'{name}__{x}_{y}')))
            if len(pending) == args.batch_size:
                flush()
        if pending:
            flush()

        kept = merge_overlaps(items, args.dup_iou)
        # tile pixel coordinates -> map coordinates
        t = src.transform
        matrix = [t.a, t.b, t.d, t.e, t.c, t.f]
        for it in kept:
            it['geometry'] = affinity.affine_transform(it['geometry'], matrix)
        gdf = gpd.GeoDataFrame([dict(score=round(it['score'], 4), score_cls=round(it['score_cls'], 4),
                                     tile=name, geometry=it['geometry']) for it in kept],
                               geometry='geometry', crs=crs) if kept else \
            gpd.GeoDataFrame(columns=['score', 'score_cls', 'tile', 'geometry'], geometry='geometry', crs=crs)
        gdf = gdf[gdf.area >= args.min_area].reset_index(drop=True)
        gdf.insert(0, 'id', range(1, len(gdf) + 1))
        gdf.insert(3, 'area_m2', gdf.area.round(1))
        if args.format == 'shp':
            gdf.to_file(osp.join(args.out, f'{name}.shp'), driver='ESRI Shapefile')
        else:
            gdf.to_file(osp.join(args.out, f'{name}.gpkg'), layer='buildings', driver='GPKG')

        mask = rasterize(((g, 1) for g in gdf.geometry), out_shape=(src.height, src.width), transform=t,
                         fill=0, dtype='uint8') if len(gdf) else np.zeros((src.height, src.width), np.uint8)
        profile = dict(driver='GTiff', width=src.width, height=src.height, count=1, dtype='uint8', crs=crs,
                       transform=t, compress='deflate', tiled=True, nodata=None)
        with rasterio.open(osp.join(args.out, f'{name}_mask.tif'), 'w', **profile) as dst:
            dst.write(mask, 1)
    return len(windows), len(items), len(gdf)


def main():
    args = parse_args()
    register_all_modules()
    tifs = find_tifs(args.src)
    print(f'{len(tifs)} tile(s) to predict')
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.config, args.ckpt_path, device)
    packer = ChipPacker()
    for i, tif in enumerate(tifs, 1):
        name = osp.splitext(osp.basename(tif))[0]
        print(f'[{i}/{len(tifs)}] {tif}')
        if args.skip_existing and osp.exists(osp.join(args.out, f'{name}.{args.format}')):
            print(f'{name}: exists, skipped')
            continue
        start = time.time()
        n_chips, n_raw, n_final = predict_tile(tif, args, model, packer, device)
        print(f'{name}: {n_chips} chips, {n_raw} raw polygons -> {n_final} buildings ({time.time() - start:.0f}s)')


if __name__ == '__main__':
    main()
