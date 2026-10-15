"""Clean the polygon outputs of predict_tiles.py without re-running the model.

Per tile <tile>.shp (or .gpkg), with its <tile>_mask*.tif next to it or in a --grid folder (the source
<tile>.tif works too, it only provides the pixel grid), this drops
    - chip-fill artifacts, where the model labels a field or road as one building up to the chip border:
      polygons larger than --max_chip_frac of a chip, polygons as long as a whole chip, polygons nearly that long running along chip borders, long
      thin strips (roads) and large triangles
    - polygons with score_cls below --score_thr, area outside --min_area/--max_area, or area above
      --big_area with score_cls below --big_score_thr
    - near-duplicates: of two polygons overlapping by more than --dup_frac of the smaller one,
      the lower-scoring one
then renumbers id, recomputes area_m2 and rewrites the mask on the original mask's grid.
Outputs go to --out (never over the inputs): <tile>.shp/.gpkg and <tile><--mask_suffix>.tif.

Example:
    python clean_predictions.py --src C:/Users/kavramova/Downloads/predictions1 --out C:/Users/kavramova/Downloads/predictions1_clean
    python clean_predictions.py --src predictions/ --out predictions_clean/ --score_thr 0.3 --min_area 10
"""
import argparse
import glob
import os
import os.path as osp

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely import STRtree
from shapely.geometry import LineString
from shapely.ops import unary_union

from make_chips import chip_offsets


def parse_args():
    parser = argparse.ArgumentParser(description='Clean predict_tiles.py outputs')
    parser.add_argument('--src', required=True, help='folder with <tile>.shp/.gpkg and <tile>_mask*.tif')
    parser.add_argument('--out', required=True, help='output folder, must differ from --src')
    parser.add_argument('--grid', nargs='*', default=[],
                        help='folder(s) with the source orthophotos <tile>.tif or masks, for tiles whose mask '
                             'is not in --src')
    parser.add_argument('--mask_suffix', default='_mask_buildings', help='output masks are <tile><suffix>.tif')
    parser.add_argument('--chip_size', default=512, type=int, help='chip size in pixels used for prediction')
    parser.add_argument('--overlap', default=64, type=int, help='chip overlap in pixels used for prediction')
    parser.add_argument('--max_chip_frac', default=0.35, type=float,
                        help='drop polygons larger than this fraction of a chip (whole-patch errors)')
    parser.add_argument('--edge_tol', default=16, type=int,
                        help='drop polygons whose longest bounding box side is within this many pixels of the chip size')
    parser.add_argument('--border_span', default=0.9, type=float,
                        help='drop polygons at least this long (fraction of the chip) ...')
    parser.add_argument('--border_len', default=0.5, type=float,
                        help='... whose outline runs along chip borders for at least this fraction of the chip')
    parser.add_argument('--strip_span', default=0.75, type=float,
                        help='drop polygons at least this long (fraction of the chip) ...')
    parser.add_argument('--min_width', default=10.0, type=float,
                        help='... whose mean width (area / longest side) is below this, in m (roads)')
    parser.add_argument('--triangle_span', default=0.5, type=float,
                        help='drop triangles (after 4 px simplification) at least this long (fraction of the chip)')
    parser.add_argument('--score_thr', default=0.3, type=float, help='minimum score_cls')
    parser.add_argument('--min_area', default=4.0, type=float, help='minimum area in m2')
    parser.add_argument('--max_area', default=None, type=float, help='drop polygons larger than this, in m2')
    parser.add_argument('--big_area', default=None, type=float,
                        help='polygons larger than this, in m2, need at least --big_score_thr')
    parser.add_argument('--big_score_thr', default=0.8, type=float, help='minimum score_cls for --big_area polygons')
    parser.add_argument('--dup_frac', default=0.8, type=float,
                        help='overlap (fraction of the smaller polygon) above which the lower-scoring one is dropped')
    args = parser.parse_args()
    if osp.abspath(args.src) == osp.abspath(args.out):
        parser.error('--out must differ from --src')
    return args


def find_grid(name, dirs):
    """The tile's mask (<name>_mask*.tif) or orthophoto (<name>.tif) in the first folder that has one."""
    for d in dirs:
        found = sorted(glob.glob(osp.join(d, f'{name}_mask*.tif'))) + glob.glob(osp.join(d, f'{name}.tif'))
        if found:
            return found[0]
    return None


def chip_borders(width, height, size, overlap, tol=1.5):
    """All chip border lines of the tile in pixel coordinates, buffered by tol pixels."""
    xs = {o for x in chip_offsets(width, size, overlap) for o in (x, x + size)}
    ys = {o for y in chip_offsets(height, size, overlap) for o in (y, y + size)}
    lines = [LineString([(x, 0), (x, height)]) for x in xs] + [LineString([(0, y), (width, y)]) for y in ys]
    return unary_union(lines).buffer(tol)


def chip_artifacts(gdf, transform, width, height, args):
    """Pixel-space tests; returns a boolean Series of polygons to drop."""
    inv = ~transform
    px_geoms = gdf.geometry.affine_transform([inv.a, inv.b, inv.d, inv.e, inv.c, inv.f])
    b = px_geoms.bounds
    span = np.maximum(b.maxx - b.minx, b.maxy - b.miny)
    size = args.chip_size
    patch = px_geoms.area > args.max_chip_frac * size ** 2
    full = span >= size - args.edge_tol
    strip = (span >= args.strip_span * size) & (gdf.area / (span * abs(transform.a)) < args.min_width)
    triangle = (span >= args.triangle_span * size) & \
        (px_geoms.simplify(4).apply(lambda g: len(g.exterior.coords)) <= 4)
    border = pd.Series(False, index=gdf.index)
    cand = (span >= args.border_span * size) & ~full & ~strip & ~triangle
    if cand.any():
        borders = chip_borders(width, height, size, args.overlap)
        border[cand] = [g.exterior.intersection(borders).length >= args.border_len * size for g in px_geoms[cand]]
    return patch | full | strip | triangle | border


def near_duplicates(gdf, dup_frac):
    """Index labels of the lower-scoring polygon of each pair overlapping more than dup_frac of the smaller."""
    geoms = gdf.geometry.values
    order = np.argsort(-gdf['score'].values, kind='stable')
    tree = STRtree(geoms)
    kept, drop = [], set()
    for i in order:
        g = geoms[i]
        if any(j in kept and g.intersection(geoms[j]).area / max(min(g.area, geoms[j].area), 1e-6) > dup_frac
               for j in tree.query(g)):
            drop.add(gdf.index[i])
        else:
            kept.append(i)
    return drop


def clean_tile(path, args):
    name = osp.splitext(osp.basename(path))[0]
    grid_path = find_grid(name, [args.src] + args.grid)
    if grid_path is None:
        print(f'{name}: SKIPPED, no {name}_mask*.tif or {name}.tif in --src or --grid')
        return
    gdf = gpd.read_file(path)
    with rasterio.open(grid_path) as src:
        profile = dict(driver='GTiff', width=src.width, height=src.height, count=1, dtype='uint8',
                       crs=src.crs or gdf.crs, transform=src.transform, compress='deflate', tiled=True, nodata=None)
    n_in = len(gdf)

    artifact = chip_artifacts(gdf, profile['transform'], profile['width'], profile['height'], args)
    low = (gdf['score_cls'] < args.score_thr) | (gdf.area < args.min_area)
    if args.max_area is not None:
        low |= gdf.area > args.max_area
    if args.big_area is not None:
        low |= (gdf.area > args.big_area) & (gdf['score_cls'] < args.big_score_thr)
    artifact_area = gdf.area[artifact].sum()
    gdf = gdf[~artifact & ~low]
    dups = near_duplicates(gdf, args.dup_frac)
    gdf = gdf.drop(index=list(dups)).reset_index(drop=True)

    gdf['id'] = range(1, len(gdf) + 1)
    gdf['area_m2'] = gdf.area.round(1)
    out_path = osp.join(args.out, osp.basename(path))
    if path.endswith('.gpkg'):
        gdf.to_file(out_path, layer='buildings', driver='GPKG')
    else:
        gdf.to_file(out_path, driver='ESRI Shapefile', encoding='utf-8')

    mask = rasterize(((g, 1) for g in gdf.geometry), out_shape=(profile['height'], profile['width']),
                     transform=profile['transform'], fill=0, dtype='uint8') if len(gdf) else \
        np.zeros((profile['height'], profile['width']), np.uint8)
    with rasterio.open(osp.join(args.out, f'{name}{args.mask_suffix}.tif'), 'w', **profile) as dst:
        dst.write(mask, 1)

    print(f'{name}: {n_in} -> {len(gdf)} polygons  (chip artifacts {int(artifact.sum())} '
          f'[{artifact_area:.0f} m2], score/area filters {int((low & ~artifact).sum())}, near-duplicates {len(dups)})')


def main():
    args = parse_args()
    paths = sorted(glob.glob(osp.join(args.src, '*.shp')) + glob.glob(osp.join(args.src, '*.gpkg')))
    if not paths:
        raise FileNotFoundError(f'no .shp or .gpkg in {args.src}')
    os.makedirs(args.out, exist_ok=True)
    for path in paths:
        clean_tile(path, args)


if __name__ == '__main__':
    main()
