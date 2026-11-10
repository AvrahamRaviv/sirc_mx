"""DOF edge KPI: sum of |diff L| on the edge pixels of each frame.

diff L is the luma difference between the warped target and the reference,
written by the DOF owner's tool. Errors near edges are the most sensitive to
motion and the ones quantization hurts most, so the KPI is: find the edge
pixels of the image, widen them into a band, and sum |diff L| over that band.

    python tools/dof_edge_kpi.py --images REF_DIR --diffs DIFF_DIR --out kpi.csv

Pairs are matched by file stem (img_001.png with img_001.npy). One CSV row per
pair, then a TOTAL row built from the summed counts (never an average of the
per-image means), which is also printed.

Image formats: .png .jpg .jpeg .tif .tiff, 8 or 16 bit, gray or color.
Color is reduced to luma with BT.601: Y = 0.299 R + 0.587 G + 0.114 B.

diff L formats (load_diff_raw, the one place to extend):
  .npy            any real dtype, one channel
  .png .tif .tiff 8 or 16 bit, one channel
--diff-offset is subtracted from png/tif values first, for a signed diff
stored with an offset (e.g. 128); abs() is taken after the offset. A .npy is
already signed and never gets the offset.

Plain |diff| png or signed with an offset? Each run prints the stored values'
min, max and median. A plain |diff| has min 0 and a median near 0. A signed
diff stored with an offset has its median near the offset (128 for 8 bit,
32768 for 16 bit) and values on both sides of it: rerun with
--diff-offset 128 (or 32768). If unsure, run once on one pair and look.

Edge detectors:
  canny (default) Gaussian blur (--sigma), then cv2.Canny on the 8-bit luma.
                  Thresholds: --canny-low/--canny-high, or per image from the
                  median m of the blurred luma: 0.66 m and 1.33 m. The median
                  rule degrades on frames that are mostly black or white
                  (m near 0 or 255): give fixed thresholds there.
  sobel           Gaussian blur, Sobel gradient magnitude, edges are the
                  pixels above its --sobel-percentile (and above 0), as a
                  second classic detector to cross-check Canny.
--dilate N widens each edge into a (2N+1)x(2N+1) square band ("near edges").

Needs OpenCV: pip install opencv-python-headless (or `pip install .[eval]`).
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:                         # optional "eval" extra
    cv2 = None

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DIFF_EXTS = (".npy", ".png", ".tif", ".tiff")
OFFSET_EXTS = (".png", ".tif", ".tiff")      # integer storage: --diff-offset applies
BT601 = (0.114, 0.587, 0.299)               # weights for B, G, R (OpenCV order)
CANNY_LOW_FACTOR, CANNY_HIGH_FACTOR = 0.66, 1.33
CSV_FIELDS = ("name", "H", "W", "edge_pixels", "edge_fraction",
              "sum_abs_diff_on_edges", "mean_abs_diff_on_edges",
              "sum_abs_diff_all", "ratio_edge_to_all")


class KpiError(ValueError):
    """A bad input the user must fix; main prints it and exits 2."""


# ---------------------------------------------------------------- loading

def _imread(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise KpiError(f"{path}: cannot be read as an image")
    return img


def load_luma(path):
    """Luma of an 8/16-bit gray or color image, as float32 in [0, 1]."""
    img = _imread(path)
    if img.dtype == np.uint8:
        full = 255.0
    elif img.dtype == np.uint16:
        full = 65535.0
    else:
        raise KpiError(f"{path}: image dtype {img.dtype}, expected 8 or 16 bit")
    img = img.astype(np.float64) / full
    if img.ndim == 3:
        if img.shape[2] not in (3, 4):
            raise KpiError(f"{path}: {img.shape[2]} channels, expected 1, 3 or 4")
        b, g, r = BT601
        img = b * img[..., 0] + g * img[..., 1] + r * img[..., 2]   # alpha dropped
    return img.astype(np.float32)


def load_diff_raw(path):
    """diff L values of one file as stored (no offset, no abs), float64.
    Add a new format here."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        diff = np.load(path, allow_pickle=False)
        if not np.issubdtype(diff.dtype, np.number) or np.iscomplexobj(diff):
            raise KpiError(f"{path}: .npy dtype {diff.dtype}, expected a real number")
    else:
        diff = _imread(path)
        if diff.dtype not in (np.uint8, np.uint16):
            raise KpiError(f"{path}: diff dtype {diff.dtype}, expected 8 or 16 bit")
    diff = np.squeeze(diff).astype(np.float64)
    if diff.ndim != 2:
        raise KpiError(f"{path}: diff shape {diff.shape}, expected one channel (H, W)")
    if not np.all(np.isfinite(diff)):
        raise KpiError(f"{path}: diff holds NaN or inf")
    return diff


def abs_diff(raw, offset=None):
    """|raw - offset|: the offset comes off before abs."""
    return np.abs(raw - float(offset)) if offset is not None else np.abs(raw)


class RawRange:
    """Range of the stored diff values over a run, to pick --diff-offset."""

    def __init__(self):
        self.lo, self.hi, self.medians = math.inf, -math.inf, []

    def update(self, raw):
        self.lo = min(self.lo, float(raw.min()))
        self.hi = max(self.hi, float(raw.max()))
        self.medians.append(float(np.median(raw)))

    def line(self, offset):
        return (f"diff L as stored: min {self.lo:.6g}, max {self.hi:.6g}, "
                f"median of per-file medians {float(np.median(self.medians)):.6g}"
                f" | --diff-offset {'none' if offset is None else f'{offset:g}'}")


# ---------------------------------------------------------------- pairing

def _stems(folder, exts, label):
    """{stem: path} of the files with an accepted extension; others ignored."""
    if not os.path.isdir(folder):
        raise KpiError(f"--{label}: {folder} is not a folder")
    found, ignored = {}, []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.startswith(".") or not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() not in exts:
            ignored.append(name)
            continue
        if stem in found:
            raise KpiError(f"--{label}: two files for {stem!r}: "
                           f"{os.path.basename(found[stem])} and {name}")
        found[stem] = path
    if ignored:
        print(f"warning: --{label}: ignored {len(ignored)} file(s) of another "
              f"type: {', '.join(ignored)}", file=sys.stderr)
    return found


def pair_files(images_dir, diffs_dir, allow_unpaired=False):
    """[(stem, image_path, diff_path)] sorted by stem; unpaired listed."""
    images = _stems(images_dir, IMAGE_EXTS, "images")
    diffs = _stems(diffs_dir, DIFF_EXTS, "diffs")
    no_diff = sorted(set(images) - set(diffs))
    no_image = sorted(set(diffs) - set(images))
    for label, stems in (("image without a diff", no_diff),
                         ("diff without an image", no_image)):
        for stem in stems:
            print(f"{label}: {stem}", file=sys.stderr)
    if (no_diff or no_image) and not allow_unpaired:
        raise KpiError(f"{len(no_diff)} image(s) without a diff and "
                       f"{len(no_image)} diff(s) without an image (listed "
                       f"above); pass --allow-unpaired to skip them")
    pairs = [(s, images[s], diffs[s]) for s in sorted(set(images) & set(diffs))]
    if not pairs:
        raise KpiError(f"no (image, diff) pairs in {images_dir} and {diffs_dir}")
    return pairs


# ---------------------------------------------------------------- edges

def blur(luma, sigma):
    """Gaussian blur, kernel size from sigma; sigma 0 means none."""
    if sigma <= 0:
        return luma
    return cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT_101)


def canny_thresholds(luma8, low=None, high=None):
    """Fixed thresholds, or 0.66 / 1.33 x the median of the 8-bit luma."""
    if low is not None and high is not None:
        return float(low), float(high)
    median = float(np.median(luma8))
    return (max(0.0, CANNY_LOW_FACTOR * median),
            min(255.0, CANNY_HIGH_FACTOR * median))


def canny_mask(luma, sigma, low=None, high=None):
    """Canny edges (bool) on the blurred luma, reduced to 8 bit as Canny needs."""
    luma8 = np.clip(np.rint(blur(luma, sigma) * 255.0), 0, 255).astype(np.uint8)
    lo, hi = canny_thresholds(luma8, low, high)
    return cv2.Canny(luma8, lo, hi, apertureSize=3, L2gradient=True) > 0


def sobel_mask(luma, sigma, percentile):
    """Pixels whose Sobel gradient magnitude is above its percentile and > 0."""
    smooth = blur(luma, sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    # Float noise of a flat frame is far below any real step (1/255 of range).
    mag[mag < 1e-6] = 0.0
    return (mag > np.percentile(mag, percentile)) & (mag > 0)


def dilate_mask(mask, pixels):
    """Widen a bool mask by `pixels` in every direction (square band)."""
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def edge_mask(luma, args):
    if args.detector == "canny":
        mask = canny_mask(luma, args.sigma, args.canny_low, args.canny_high)
    else:
        mask = sobel_mask(luma, args.sigma, args.sobel_percentile)
    return dilate_mask(mask, args.dilate)


# ---------------------------------------------------------------- KPI

def _ratio(num, den):
    return num / den if den else math.nan


def kpi_row(name, mask, abs_dl):
    """One CSV row from an edge mask and |diff L| of the same shape."""
    h, w = mask.shape
    edge_pixels = int(mask.sum())
    on_edges = float(abs_dl[mask].sum())
    total = float(abs_dl.sum())
    return {"name": name, "H": h, "W": w, "edge_pixels": edge_pixels,
            "edge_fraction": edge_pixels / (h * w),
            "sum_abs_diff_on_edges": on_edges,
            "mean_abs_diff_on_edges": _ratio(on_edges, edge_pixels),
            "sum_abs_diff_all": total,
            "ratio_edge_to_all": _ratio(on_edges, total)}


def total_row(rows):
    """Sums of the counts and sums; the means recomputed from them."""
    pixels = sum(r["H"] * r["W"] for r in rows)
    edge_pixels = sum(r["edge_pixels"] for r in rows)
    on_edges = sum(r["sum_abs_diff_on_edges"] for r in rows)
    total = sum(r["sum_abs_diff_all"] for r in rows)
    return {"name": "TOTAL", "H": "", "W": "", "edge_pixels": edge_pixels,
            "edge_fraction": edge_pixels / pixels,
            "sum_abs_diff_on_edges": on_edges,
            "mean_abs_diff_on_edges": _ratio(on_edges, edge_pixels),
            "sum_abs_diff_all": total,
            "ratio_edge_to_all": _ratio(on_edges, total)}


def measure_pair(stem, image_path, diff_path, args, raw_range):
    luma = load_luma(image_path)
    raw = load_diff_raw(diff_path)
    raw_range.update(raw)
    if luma.shape != raw.shape:
        raise KpiError(f"{stem}: image {os.path.basename(image_path)} is "
                       f"{luma.shape[0]}x{luma.shape[1]} but diff "
                       f"{os.path.basename(diff_path)} is "
                       f"{raw.shape[0]}x{raw.shape[1]}")
    mask = edge_mask(luma, args)
    if args.save_masks:
        out = os.path.join(args.save_masks, f"{stem}.png")
        if not cv2.imwrite(out, mask.astype(np.uint8) * 255):
            raise KpiError(f"{out}: could not write the mask")
    stored_as_int = os.path.splitext(diff_path)[1].lower() in OFFSET_EXTS
    offset = args.diff_offset if stored_as_int else None
    return kpi_row(stem, mask, abs_diff(raw, offset))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- CLI

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", required=True,
                    help="folder of images to find edges on (the reference frames)")
    ap.add_argument("--diffs", required=True, help="folder of diff L files")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--allow-unpaired", action="store_true",
                    help="skip (still list) files with no partner instead of failing")
    ap.add_argument("--diff-offset", type=float, default=None,
                    help="subtracted from png/tif diff values before abs "
                         "(signed diff stored with an offset, e.g. 128 or "
                         "32768); default none. The run prints the stored "
                         "min/max/median to tell which one a file is")
    ap.add_argument("--detector", choices=("canny", "sobel"), default="canny",
                    help="edge detector (default canny)")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian blur sigma before edges, 0 = none (default 1.0)")
    ap.add_argument("--canny-low", type=float, default=None,
                    help="fixed Canny low threshold (with --canny-high); "
                         "default: 0.66 x median per image")
    ap.add_argument("--canny-high", type=float, default=None,
                    help="fixed Canny high threshold (with --canny-low); "
                         "default: 1.33 x median per image")
    ap.add_argument("--sobel-percentile", type=float, default=90.0,
                    help="sobel: edges are above this percentile of the "
                         "gradient magnitude (default 90)")
    ap.add_argument("--dilate", type=int, default=1,
                    help="widen edges by N pixels each side, 0 = the line only (default 1)")
    ap.add_argument("--save-masks", default=None,
                    help="folder to write each edge mask as <stem>.png (0/255)")
    return ap


def check_args(ap, args):
    if (args.canny_low is None) != (args.canny_high is None):
        ap.error("--canny-low and --canny-high go together")
    if args.canny_low is not None and not 0 <= args.canny_low <= args.canny_high:
        ap.error("need 0 <= --canny-low <= --canny-high")
    if args.sigma < 0:
        ap.error("--sigma must be >= 0")
    if not 0 <= args.sobel_percentile < 100:
        ap.error("--sobel-percentile must be in [0, 100)")
    if args.dilate < 0:
        ap.error("--dilate must be >= 0")


def format_total(row):
    return (f"TOTAL | edge_pixels {row['edge_pixels']} "
            f"({row['edge_fraction']:.4%}) | sum |dL| on edges "
            f"{row['sum_abs_diff_on_edges']:.6g} | mean on edges "
            f"{row['mean_abs_diff_on_edges']:.6g} | sum |dL| all "
            f"{row['sum_abs_diff_all']:.6g} | edge/all "
            f"{row['ratio_edge_to_all']:.4f}")


def run(args):
    if cv2 is None:
        raise KpiError("OpenCV is missing: pip install opencv-python-headless")
    pairs = pair_files(args.images, args.diffs, args.allow_unpaired)
    if args.save_masks:
        os.makedirs(args.save_masks, exist_ok=True)
    raw_range = RawRange()
    rows = [measure_pair(stem, img, diff, args, raw_range)
            for stem, img, diff in pairs]
    for r in rows:
        if r["edge_pixels"] == 0:
            print(f"note: {r['name']}: no edge pixels, its mean is NaN",
                  file=sys.stderr)
    total = total_row(rows)
    write_csv(args.out, rows + [total])
    print(f"{len(rows)} pairs -> {args.out}")
    print(raw_range.line(args.diff_offset))
    print(format_total(total))
    return total


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    check_args(ap, args)
    try:
        run(args)
    except KpiError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
