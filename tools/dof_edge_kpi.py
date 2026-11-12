"""DOF edge KPI: sum of |diff L| on the edge pixels of each frame.

diff L is the luma difference between the warped target and the reference,
written by the DOF owner's tool. Errors near edges are the most sensitive to
motion and the ones quantization hurts most, so the KPI is: find the edge
pixels of the image, widen them into a band, and sum |diff L| over that band.

    python tools/dof_edge_kpi.py --images FRAMES_DIR --diffs DIFF_DIR --out kpi.csv

--images holds every input frame (in diff-name pairing both A and B); edges
are found on the reference one. Best: the network's Y8 inputs (one channel,
already luma, read as is).

Pairing, one of two modes:
  same stem       img_001.png with img_001.npy.
  from diff name  the diff names both frames: <A>_<B>_diff_L.<ext>, A and B
                  image stems in --images (they may hold "_": the split used
                  is the one where both halves are image stems, and none or
                  several is an error). Edges are found on the reference,
                  --ref-frame first (A, the default: DOF infer's im0, from
                  its save code and measured 2026-11-10) or second (B); each
                  run prints which. A diff whose A or B frame is missing is
                  an error (unless --allow-unpaired); frames no diff names
                  are only a note (a dataset folder always has some). Used
                  when every diff stem ends in _diff_L, or forced by
                  --pair-from-diff-name.
One CSV row per pair (named by the diff stem), then a TOTAL row built from the
summed counts (never an average of the per-image means), which is also printed.

Float vs quant (opt-in): --diffs-float DIR holds the diff L files of the float
model on the same frames; --diffs stays the run under test (quant).

    python tools/dof_edge_kpi.py --images FRAMES_DIR --diffs QUANT_DIR
                                 --diffs-float FLOAT_DIR --out kpi.csv

Both folders go through the same loader and pairing and must hold the same
pairs (one run only: an error naming it, or with --allow-unpaired skipped in
both and counted in a note). Edges are found once per pair, on the reference
frame, and the same band is used for every measure. Each row adds:
  float_*  the float run's KPI on that band
  delta_*  quant - float of each measure: what quantization adds over float
  qerr_*   the KPI of |dq - df|. diff L = warped - reference and the reference
           is the same in both runs, so for SIGNED diffs (.npy, or with an
           offset) dq - df = warped_quant - warped_float: the quantization
           error alone, the float model's own error cancelled.
  diff_kind  signed, or abs when a file of the pair is a plain |diff| (no
           offset): qerr_* is then not that difference and only delta_* is
           meaningful; the run warns. TOTAL: signed, abs or mixed.
--diff-offset-float is the float run's offset (default: --diff-offset;
"none" for a plain |diff| float run against a signed quant one). The
two files of a pair must have one shape, and one stored dtype when both are
.npy or both are images (8 vs 16 bit is an error; npy vs png is allowed).

Image formats: .png .bmp .jpg .jpeg .tif .tiff, 8 or 16 bit, gray or color.
A one-channel image is used as is; color is reduced to luma with BT.601:
Y = 0.299 R + 0.587 G + 0.114 B.

diff L formats (read_diff, the one place to extend):
  .npy            any real dtype
  .png .tif .tiff 8 or 16 bit
  .jpg .jpeg      8 bit. JPEG is lossy: the diff carries compression noise of
                  a few levels everywhere. The KPI compares runs on the same
                  data, so it stays fair, but png is cleaner when available.
A diff with 3 channels is taken as gray when its channels agree within
--diff-channel-tol (default 2, for JPEG): channel 0 is used. Otherwise it is an
error naming the largest channel difference (a colormap needs another rule);
a color diff is never turned into luma.
--diff-offset is subtracted from png/tif/jpg values first, for a signed diff
stored with an offset (e.g. 128); abs() is taken after the offset. A .npy is
already signed and never gets the offset. Both offset flags take a finite
number or "none" (no offset; --diff-offset's default); nan and inf are refused.

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

Motion vectors (opt-in): --mv QUANT_MV_DIR --mv-float FLOAT_MV_DIR, DOF
infer's <output_folder>/mv/ of each run (.dof files, named <A>_<B>.dof or
<A>_<B>_<suffix>.dof: the diff's pair, split as the diff names are). Per pair
and in TOTAL, on the same band and on all pixels: endpoint error
EPE = sqrt((uq - uf)^2 + (vq - vf)^2), its sum and mean, the fraction of
pixels with EPE > 0.5 and > 1, the float flow's mean |flow| and the unit-free
EPE / |flow| (sum over sum). A pair needs both mv files (else an error, or
with --allow-unpaired skipped in every measure).
The .dof layout is flags (read_mv, the one place to change it), checked
against the file size exactly; a mismatch prints the sizes that would fit.
Defaults are DOF infer's files as measured on one (2026-11-12): --mv-dtype
int16 (little-endian), --mv-layout 2hw (planar: one plane, then the other),
--mv-header-bytes 0, --mv-shape 1/4 of the reference frame. --mv-scale
(pixels = value / scale) defaults to 1: raw units, with a warning, since the
fixed-point scale is not known yet. --mv-unit-res quarter|full names the grid
the vectors' pixels are on (a label; nothing is rescaled). A flow smaller than
the frame by an integer k gets the band reduced to it: a cell is on the band
when any frame pixel of its k x k block is. --mv-probe reads the first --mv
file only and prints what fits and its values; it writes nothing.

Needs OpenCV: pip install opencv-python-headless (or `pip install '.[eval]'`).
"""

import argparse
import csv
import math
import os
import sys
from typing import NamedTuple

import numpy as np

try:
    import cv2
except ImportError:                         # optional "eval" extra
    cv2 = None

IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
DIFF_EXTS = (".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
OFFSET_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")   # integer storage: --diff-offset applies
DIFF_NAME_SUFFIX = "_diff_L"
NOTE_NAMES_SHOWN = 5            # frames no diff names: how many the note lists
DIFF_CHANNEL_TOL = 2.0          # JPEG writes a gray diff as 3 channels, +-1 or 2 apart
BT601 = (0.114, 0.587, 0.299)               # weights for B, G, R (OpenCV order)
CANNY_LOW_FACTOR, CANNY_HIGH_FACTOR = 0.66, 1.33
# --diff-offset-float not given: a sentinel, as None already means "none".
SAME_AS_DIFF_OFFSET = object()
MASK_FIELDS = ("name", "H", "W", "edge_pixels", "edge_fraction")
# The measures of one |diff| on the band; float_/delta_/qerr_ prefix them.
DIFF_FIELDS = ("sum_abs_diff_on_edges", "mean_abs_diff_on_edges",
               "sum_abs_diff_all", "ratio_edge_to_all")
CSV_FIELDS = MASK_FIELDS + DIFF_FIELDS
FLOAT, DELTA, QERR = "float_", "delta_", "qerr_"
COMPARE_FIELDS = (CSV_FIELDS + tuple(p + f for p in (FLOAT, DELTA, QERR)
                                     for f in DIFF_FIELDS) + ("diff_kind",))

# Motion vectors (.dof): the layout is not known here, so every part of it is a
# flag, checked against the file size. Little-endian, as x86 and ARM write.
MV_EXTS = (".dof",)
MV_DTYPES = ("float32", "float16", "int16", "int32", "float64")
MV_LAYOUTS = ("hw2", "2hw")       # H x W x (u, v) interleaved, or the u plane then v
MV_RESOLUTIONS = (1, 2, 4, 8)     # frame size divided by these: the sizes a check offers
MV_DEFAULT_FACTOR = 4             # DOF infer's mv: 1/4 of the frame (owner's probe, 2026-11-12)
MV_UNIT_RES = {"quarter": 4, "full": 1}   # the grid the vectors' pixels are on: its factor
MV_HEADER_HINT = 4096             # a size this much above the data is offered as a header
MV_EPE_THRESHOLDS = (0.5, 1.0)    # px: the fraction of pixels with EPE above each
FLO_MAGIC = b"PIEH"               # Middlebury .flo: float 202021.25, then int32 W, H
MV_REGIONS = {"on_edges": "mv_edge_pixels", "all": "mv_pixels"}   # region: its pixel count


def mv_over(threshold):
    """The count and fraction field names of EPE > threshold, per region."""
    return f"mv_n_epe_over_{threshold:g}px", f"mv_frac_epe_over_{threshold:g}px"


def mv_region_fields(region):
    """The CSV fields of one region: sums and counts, each with its mean."""
    over = [f"{name}_{region}" for t in MV_EPE_THRESHOLDS for name in mv_over(t)]
    return (f"mv_epe_sum_{region}", f"mv_epe_mean_{region}", *over,
            f"mv_float_mag_sum_{region}", f"mv_float_mag_mean_{region}",
            f"mv_rel_epe_{region}")


def mv_region_sums(region):
    """The fields of mv_region_fields a TOTAL sums (the rest are means of them)."""
    return (f"mv_epe_sum_{region}", *[f"{mv_over(t)[0]}_{region}" for t in MV_EPE_THRESHOLDS],
            f"mv_float_mag_sum_{region}")


MV_FIELDS = (("mv_H", "mv_W", *MV_REGIONS.values())
             + tuple(f for region in MV_REGIONS for f in mv_region_fields(region)))
MV_SUMMED = (*MV_REGIONS.values(), *[f for region in MV_REGIONS for f in mv_region_sums(region)])


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


def gray_channel(diff, path, tol):
    """Channel 0 of an (H, W, C) diff whose channels agree within `tol`."""
    spread = float((diff.max(axis=2) - diff.min(axis=2)).max())
    if spread > tol:
        raise KpiError(f"{path}: diff has {diff.shape[2]} channels that differ by "
                       f"up to {spread:g} (> --diff-channel-tol {tol:g}); a color "
                       f"or colormap diff is not supported")
    return diff[..., 0]


class StoredDiff(NamedTuple):
    values: np.ndarray      # (H, W) float64, as stored: no offset, no abs
    dtype: np.dtype         # the file's own dtype, before float64


def read_diff(path, channel_tol=DIFF_CHANNEL_TOL):
    """One diff L file as a StoredDiff. Add a new format here."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        diff = np.load(path, allow_pickle=False)
        if not np.issubdtype(diff.dtype, np.number) or np.iscomplexobj(diff):
            raise KpiError(f"{path}: .npy dtype {diff.dtype}, expected a real number")
    else:
        diff = _imread(path)
        if diff.dtype not in (np.uint8, np.uint16):
            raise KpiError(f"{path}: diff dtype {diff.dtype}, expected 8 or 16 bit")
    stored_dtype = diff.dtype
    diff = diff.astype(np.float64)
    if diff.ndim == 3 and diff.shape[2] == 1:          # (H, W, 1): H or W may be 1
        diff = diff[..., 0]
    if diff.ndim == 3 and diff.shape[2] <= 4:          # channels last, as imread
        diff = gray_channel(diff, path, channel_tol)
    if diff.ndim != 2:
        raise KpiError(f"{path}: diff shape {diff.shape}, expected (H, W) "
                       f"or (H, W, channels)")
    if not np.all(np.isfinite(diff)):
        raise KpiError(f"{path}: diff holds NaN or inf")
    return StoredDiff(diff, stored_dtype)


class RunDiff(NamedTuple):
    """One run's diff L of one pair, ready to measure."""
    path: str
    values: np.ndarray      # signed when `signed`: the offset is already off
    signed: bool            # .npy, or integer storage given an offset
    dtype: np.dtype         # stored dtype
    is_image: bool          # png/tif/jpg storage (else .npy)


def read_run_diff(path, offset, channel_tol, raw_range):
    """A RunDiff: the run's offset comes off integer storage only (a .npy is
    already signed); abs is left to the measure."""
    stored = read_diff(path, channel_tol)
    raw_range.update(stored.values)
    is_image = os.path.splitext(path)[1].lower() in OFFSET_EXTS
    offset = offset if is_image else None
    values = stored.values - float(offset) if offset is not None else stored.values
    return RunDiff(path, values, not is_image or offset is not None,
                   stored.dtype, is_image)


class RawRange:
    """Range of the stored diff values over a run, to pick --diff-offset."""

    def __init__(self, label="diff L", offset_flag="--diff-offset"):
        self.lo, self.hi, self.medians = math.inf, -math.inf, []
        self.label, self.offset_flag = label, offset_flag

    def update(self, raw):
        self.lo = min(self.lo, float(raw.min()))
        self.hi = max(self.hi, float(raw.max()))
        self.medians.append(float(np.median(raw)))

    def line(self, offset):
        return (f"{self.label} as stored: min {self.lo:.6g}, max {self.hi:.6g}, "
                f"median of per-file medians {float(np.median(self.medians)):.6g}"
                f" | {self.offset_flag} {'none' if offset is None else f'{offset:g}'}")


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


class Pairing(NamedTuple):
    """One run's pairs, and what was used to pair them (the mv files pair the same way)."""
    pairs: list             # [(name, image_path, diff_path)] sorted by name
    notes: list             # note lines for the run summary
    images: dict            # {stem: path} of --images
    from_diff_name: bool    # the mode used: <A>_<B>_diff_L names, or same stem


def pair_files(images_dir, diffs_dir, allow_unpaired=False,
               from_diff_name=None, ref_frame="first", diffs_label="diffs"):
    """([(name, image_path, diff_path)] sorted by name, [note lines]): the
    pairs and notes of pair_folder."""
    pairing = pair_folder(_stems(images_dir, IMAGE_EXTS, "images"), images_dir, diffs_dir,
                          allow_unpaired, from_diff_name, ref_frame, diffs_label)
    return pairing.pairs, pairing.notes


def pair_folder(images, images_dir, diffs_dir, allow_unpaired=False,
                from_diff_name=None, ref_frame="first", diffs_label="diffs"):
    """The Pairing of one diff folder with the listed images ({stem: path}).
    Unpaired files are listed.

    from_diff_name: True / False forces a mode; None picks diff-name pairing
    when every diff stem ends in _diff_L. diffs_label: the folder's flag
    without its dashes, for messages ("diffs-float" for the float run)."""
    diffs = _stems(diffs_dir, DIFF_EXTS, diffs_label)
    # The quant run's messages read as they always have; another run names its flag.
    where = "" if diffs_label == "diffs" else f" in --{diffs_label}"
    if from_diff_name is None:
        from_diff_name = use_diff_name_pairing(diffs, diffs_label)
    if from_diff_name:
        # Edges are found on --diffs' pairs only: the float run's line says the mode.
        edges = ("" if where else f", edges on the {ref_frame} frame "
                 f"({'A' if ref_frame == 'first' else 'B'}, --ref-frame {ref_frame})")
        print(f"pairing{where}: from <A>_<B>{DIFF_NAME_SUFFIX} names{edges}")
        pairs, unnamed = pair_from_diff_name(images, diffs, ref_frame,
                                             allow_unpaired, where)
        notes = [unnamed_note(unnamed)] if unnamed else []
    else:
        pairs, notes = pair_same_stem(images, diffs, allow_unpaired, where), []
    if not pairs:
        raise KpiError(f"no (image, diff) pairs in {images_dir} and {diffs_dir}")
    return Pairing(pairs, notes, images, from_diff_name)


def listed_note(what, names):
    """A note line: how many `what`, and the first few names."""
    shown = ", ".join(names[:NOTE_NAMES_SHOWN])
    more = ", ..." if len(names) > NOTE_NAMES_SHOWN else ""
    return f"note: {len(names)} {what}: {shown}{more}"


def unnamed_note(stems):
    """The note for frames no diff names: their count and the first few."""
    return listed_note("frame(s) in --images named by no diff, not measured", stems)


def use_diff_name_pairing(diffs, diffs_label="diffs"):
    """True when every diff stem ends in _diff_L; a mix is an error."""
    named = [s for s in diffs if s.endswith(DIFF_NAME_SUFFIX)]
    if named and len(named) != len(diffs):
        other = sorted(set(diffs) - set(named))
        raise KpiError(f"--{diffs_label} mixes *{DIFF_NAME_SUFFIX} names with others "
                       f"({', '.join(other)}); pass --pair-from-diff-name or "
                       f"--pair-same-stem")
    return bool(named)


def stem_prefixes(name, stems):
    """[(stem, rest)] for every stem in `stems` that `name` is, or starts
    with followed by "_": "a_b_c" with stems {a, a_b} gives
    [("a", "b_c"), ("a_b", "c")]; rest is "" when name is the stem."""
    ends = [i for i, ch in enumerate(name) if ch == "_"] + [len(name)]
    return [(name[:i], name[i + 1:]) for i in ends if name[:i] in stems]


def name_splits(core, image_stems, with_suffix=False):
    """[(A, B, suffix)]: every way `core` is '<A>_<B>' (with_suffix: or
    '<A>_<B>_<suffix>'), A and B image stems; suffix "" when there is none."""
    return [(a, b, suffix) for a, rest in stem_prefixes(core, image_stems)
            for b, suffix in stem_prefixes(rest, image_stems)
            if with_suffix or not suffix]


def split_diff_name(stem, image_stems):
    """(A, B) of '<A>_<B>_diff_L' where both are image stems; None if none
    (or the name does not end in _diff_L)."""
    if not stem.endswith(DIFF_NAME_SUFFIX):
        return None
    core = stem[:-len(DIFF_NAME_SUFFIX)]
    splits = [(a, b) for a, b, _ in name_splits(core, image_stems)]
    if len(splits) > 1:
        raise KpiError(f"diff {stem!r}: ambiguous, several splits name two "
                       f"images: {', '.join(f'{a} + {b}' for a, b in splits)}")
    return splits[0] if splits else None


def pair_from_diff_name(images, diffs, ref_frame, allow_unpaired, where=""):
    """(pairs, frames no diff names). A diff naming no two images is listed
    and is an error unless allow_unpaired; a frame no diff names is not an
    error: a dataset folder always holds some (a sequence's ends). where:
    the folder's words in messages, "" for --diffs."""
    pairs, unsplit, named = [], [], set()
    for stem in sorted(diffs):
        split = split_diff_name(stem, images)
        if split is None:
            unsplit.append(stem)
            continue
        named.update(split)
        ref = split[0] if ref_frame == "first" else split[1]
        pairs.append((stem, images[ref], diffs[stem]))
    for stem in unsplit:
        label = ("diff whose name splits into no two images"
                 if stem.endswith(DIFF_NAME_SUFFIX)
                 else f"diff not named <A>_<B>{DIFF_NAME_SUFFIX}")
        print(f"{label}{where}: {stem}", file=sys.stderr)
    if unsplit and not allow_unpaired:
        raise KpiError(f"{len(unsplit)} diff(s){where} name no two images in --images "
                       f"(listed above); pass --allow-unpaired to skip them")
    return pairs, sorted(set(images) - named)


def pair_same_stem(images, diffs, allow_unpaired, where=""):
    """Pairs files of one stem: img_001.png with img_001.npy. where: as in
    pair_from_diff_name."""
    no_diff = sorted(set(images) - set(diffs))
    no_image = sorted(set(diffs) - set(images))
    for label, stems in (("image without a diff", no_diff),
                         ("diff without an image", no_image)):
        for stem in stems:
            print(f"{label}{where}: {stem}", file=sys.stderr)
    if (no_diff or no_image) and not allow_unpaired:
        raise KpiError(f"{len(no_diff)} image(s) without a diff{where} and "
                       f"{len(no_image)} diff(s) without an image (listed "
                       f"above); pass --allow-unpaired to skip them")
    return [(s, images[s], diffs[s]) for s in sorted(set(images) & set(diffs))]


def match_runs(quant_pairs, float_pairs, allow_unpaired):
    """([(name, image_path, quant_diff, float_diff)] sorted by name, names in
    one run only). Each run was paired on its own; a pair in one run only is
    listed and is an error unless allow_unpaired (then skipped in both)."""
    quant = {name: (image, diff) for name, image, diff in quant_pairs}
    flt = {name: diff for name, _, diff in float_pairs}
    only_quant = sorted(set(quant) - set(flt))
    only_float = sorted(set(flt) - set(quant))
    for label, names in (("pair only in --diffs", only_quant),
                         ("pair only in --diffs-float", only_float)):
        for name in names:
            print(f"{label}: {name}", file=sys.stderr)
    if (only_quant or only_float) and not allow_unpaired:
        raise KpiError(f"{len(only_quant)} pair(s) only in --diffs and "
                       f"{len(only_float)} only in --diffs-float (listed above); "
                       f"pass --allow-unpaired to skip them in both")
    both = sorted(set(quant) & set(flt))
    if not both:
        raise KpiError("no pair is in both --diffs and --diffs-float")
    return ([(name, *quant[name], flt[name]) for name in both],
            only_quant + only_float)


class MvName(NamedTuple):
    pair: str               # the name of the diff L pair it belongs to
    frames: tuple           # (A, B), or (stem,) in same-stem pairing
    suffix: str             # what follows the pair's stem(s) after "_", or ""


def split_mv_name(stem, images, from_diff_name):
    """The MvName of a .dof stem: '<A>_<B>' or '<A>_<B>_<suffix>' (A, B image
    stems, the diff L names' split) for diff-name pairing, '<stem>' or
    '<stem>_<suffix>' for same-stem pairing; None if it names no pair. Several
    readings are an error, never a guess."""
    if from_diff_name:
        names = [MvName(f"{a}_{b}{DIFF_NAME_SUFFIX}", (a, b), suffix)
                 for a, b, suffix in name_splits(stem, images, with_suffix=True)]
    else:
        names = [MvName(s, (s,), suffix) for s, suffix in stem_prefixes(stem, images)]
    if len(names) > 1:
        raise KpiError(f"mv file {stem!r}: ambiguous, it reads as the pair of "
                       f"{' or '.join(' + '.join(n.frames) for n in names)}")
    return names[0] if names else None


def index_mv(folder, label, images, from_diff_name):
    """({pair name: path}, [stems naming no pair], suffix) of one mv folder.
    Two files of one pair, or two suffixes in one folder, are an error: one
    infer run writes one naming, and a second suffix may be a misread name."""
    files = _stems(folder, MV_EXTS, label)
    if not files:
        raise KpiError(f"--{label}: no {'/'.join(MV_EXTS)} file in {folder}")
    by_pair, unnamed, suffixes = {}, [], set()
    for stem, path in sorted(files.items()):
        name = split_mv_name(stem, images, from_diff_name)
        if name is None:
            unnamed.append(stem)
            continue
        if name.pair in by_pair:
            raise KpiError(f"--{label}: two mv files for the pair {name.pair}: "
                           f"{os.path.basename(by_pair[name.pair])} and "
                           f"{os.path.basename(path)}")
        by_pair[name.pair] = path
        suffixes.add(name.suffix)
    if len(suffixes) > 1:
        raise KpiError(f"--{label}: the mv file names end in "
                       f"{len(suffixes)} suffixes "
                       f"({', '.join(repr(s) for s in sorted(suffixes))}); "
                       f"one folder holds one naming")
    return by_pair, unnamed, suffixes.pop() if suffixes else ""


class MvFiles(NamedTuple):
    """The mv files matched to the diff pairs."""
    paths: dict             # {pair name: (quant mv path, float mv path)}
    suffixes: tuple         # (--mv's, --mv-float's) name suffix, printed


def match_mv(pairs, pairing, quant_dir, float_dir, allow_unpaired):
    """(pairs that have both mv files, MvFiles, [names skipped]). An mv file
    naming no pair or a pair with no diff, and a pair missing an mv file, are
    listed and an error unless allow_unpaired (then the pair is skipped in
    every measure, so all TOTALs are over the same pairs)."""
    names = {p[0] for p in pairs}
    runs, unmatched, missing = [], 0, set()
    for label, folder in (("mv", quant_dir), ("mv-float", float_dir)):
        by_pair, unnamed, suffix = index_mv(folder, label, pairing.images,
                                            pairing.from_diff_name)
        no_diff = sorted(os.path.basename(by_pair[n]) for n in set(by_pair) - names)
        no_mv = sorted(names - set(by_pair))
        for what, listed in ((f"mv file in --{label} naming no pair", unnamed),
                             (f"mv file in --{label} of a pair with no diff", no_diff),
                             (f"pair without an mv file in --{label}", no_mv)):
            for name in listed:
                print(f"{what}: {name}", file=sys.stderr)
        unmatched += len(unnamed) + len(no_diff)
        missing.update(no_mv)
        runs.append((by_pair, suffix))
    if (unmatched or missing) and not allow_unpaired:
        raise KpiError(f"{unmatched} mv file(s) match no diff pair and {len(missing)} "
                       f"pair(s) miss an mv file (listed above); pass "
                       f"--allow-unpaired to skip them")
    kept = [p for p in pairs if p[0] not in missing]
    if not kept:
        raise KpiError("no pair has both its --mv and its --mv-float file")
    (quant, quant_suffix), (flt, float_suffix) = runs
    paths = {p[0]: (quant[p[0]], flt[p[0]]) for p in kept}
    return kept, MvFiles(paths, (quant_suffix, float_suffix)), sorted(missing)


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


def measures(on_edges, total, edge_pixels, prefix=""):
    """The DIFF_FIELDS from two sums and the band size, keys prefixed."""
    values = (on_edges, _ratio(on_edges, edge_pixels), total, _ratio(on_edges, total))
    return {prefix + field: value for field, value in zip(DIFF_FIELDS, values)}


def diff_measures(mask, abs_dl, prefix=""):
    """The measures of one |diff| (same shape as the mask) on the band."""
    return measures(float(abs_dl[mask].sum()), float(abs_dl.sum()),
                    int(mask.sum()), prefix)


def total_measures(rows, edge_pixels, prefix=""):
    """The measures of the summed sums: means recomputed, never averaged."""
    return measures(sum(r[prefix + "sum_abs_diff_on_edges"] for r in rows),
                    sum(r[prefix + "sum_abs_diff_all"] for r in rows),
                    edge_pixels, prefix)


def delta_measures(row):
    """quant - float of each measure of a row (the TOTAL row too)."""
    return {DELTA + f: row[f] - row[FLOAT + f] for f in DIFF_FIELDS}


def kpi_row(name, mask, abs_dl):
    """One CSV row from an edge mask and |diff L| of the same shape."""
    h, w = mask.shape
    edge_pixels = int(mask.sum())
    return {"name": name, "H": h, "W": w, "edge_pixels": edge_pixels,
            "edge_fraction": edge_pixels / (h * w),
            **diff_measures(mask, abs_dl)}


def total_row(rows):
    """Sums of the counts and sums; the means recomputed from them."""
    pixels = sum(r["H"] * r["W"] for r in rows)
    edge_pixels = sum(r["edge_pixels"] for r in rows)
    return {"name": "TOTAL", "H": "", "W": "", "edge_pixels": edge_pixels,
            "edge_fraction": edge_pixels / pixels,
            **total_measures(rows, edge_pixels)}


def compare_row(name, mask, quant, flt):
    """kpi_row of the quant run plus the float run, delta and |dq - df| on
    the same band. dq - df is warped_quant - warped_float only when both
    diffs are signed: diff_kind says which case the row is."""
    row = kpi_row(name, mask, np.abs(quant.values))
    row.update(diff_measures(mask, np.abs(flt.values), FLOAT))
    row.update(delta_measures(row))
    row.update(diff_measures(mask, np.abs(quant.values - flt.values), QERR))
    row["diff_kind"] = "signed" if quant.signed and flt.signed else "abs"
    return row


def compare_total_row(rows):
    """total_row plus every compare measure from the summed sums."""
    total = total_row(rows)
    for prefix in (FLOAT, QERR):
        total.update(total_measures(rows, total["edge_pixels"], prefix))
    total.update(delta_measures(total))
    kinds = {r["diff_kind"] for r in rows}
    total["diff_kind"] = kinds.pop() if len(kinds) == 1 else "mixed"
    return total


def check_shape(stem, image_path, luma, diff):
    if luma.shape != diff.values.shape:
        raise KpiError(f"{stem}: image {os.path.basename(image_path)} is "
                       f"{luma.shape[0]}x{luma.shape[1]} but diff "
                       f"{os.path.basename(diff.path)} is "
                       f"{diff.values.shape[0]}x{diff.values.shape[1]}")


def check_runs_match(stem, quant, flt):
    """One shape for both runs' files, and one stored dtype when both are
    images or both .npy (8 vs 16 bit is another scale); npy vs png is fine."""
    q_name, f_name = os.path.basename(quant.path), os.path.basename(flt.path)
    if quant.values.shape != flt.values.shape:
        raise KpiError(f"{stem}: --diffs {q_name} is {quant.values.shape[0]}x"
                       f"{quant.values.shape[1]} but --diffs-float {f_name} is "
                       f"{flt.values.shape[0]}x{flt.values.shape[1]}")
    if quant.is_image == flt.is_image and quant.dtype != flt.dtype:
        raise KpiError(f"{stem}: --diffs {q_name} is {quant.dtype} but "
                       f"--diffs-float {f_name} is {flt.dtype}")


def reference_mask(stem, luma, args):
    """The band of the reference frame, saved if asked."""
    mask = edge_mask(luma, args)
    if args.save_masks:
        out = os.path.join(args.save_masks, f"{stem}.png")
        if not cv2.imwrite(out, mask.astype(np.uint8) * 255):
            raise KpiError(f"{out}: could not write the mask")
    return mask


def measure_pair(stem, image_path, diff_path, args, ranges, mv_paths=None):
    """kpi_row of one pair; ranges[0] is the run's RawRange. mv_paths: the
    pair's (quant, float) mv files, measured on the same band."""
    luma = load_luma(image_path)
    diff = read_run_diff(diff_path, args.diff_offset, args.diff_channel_tol, ranges[0])
    check_shape(stem, image_path, luma, diff)
    mask = reference_mask(stem, luma, args)
    return with_mv(kpi_row(stem, mask, np.abs(diff.values)), mask, mv_paths, args)


def measure_compare(stem, image_path, quant_path, float_path, args, ranges, mv_paths=None):
    """compare_row of one pair; ranges: the (quant, float) RawRange. mv_paths:
    as in measure_pair."""
    luma = load_luma(image_path)
    quant = read_run_diff(quant_path, args.diff_offset, args.diff_channel_tol, ranges[0])
    check_shape(stem, image_path, luma, quant)
    flt = read_run_diff(float_path, float_offset(args), args.diff_channel_tol, ranges[1])
    check_runs_match(stem, quant, flt)
    mask = reference_mask(stem, luma, args)
    return with_mv(compare_row(stem, mask, quant, flt), mask, mv_paths, args)


# ---------------------------------------------------------------- motion vectors

class MvFormat(NamedTuple):
    """How a .dof file is stored: the --mv-* flags, defaults resolved."""
    dtype: str              # one of MV_DTYPES, little-endian
    layout: str             # one of MV_LAYOUTS
    header_bytes: int       # skipped at the start of the file
    scale: float            # pixels = stored value / scale
    scale_given: bool       # False: scale 1, the values are raw units
    shape: tuple            # (H, W) of the flow, or None: frame / MV_DEFAULT_FACTOR
    unit_res: str           # the grid of the vectors' pixels: a MV_UNIT_RES key


def mv_data_bytes(shape, dtype):
    """Bytes of an (H, W) flow of two values per pixel in `dtype`."""
    return shape[0] * shape[1] * 2 * np.dtype(dtype).itemsize


def read_mv_values(path, dtype, header_bytes):
    """Every value after the header, as float64 (nothing checked yet: read_mv
    refuses NaN/inf by name, the probe counts them, so numpy's own warning on
    casting one is only noise)."""
    with open(path, "rb") as f:
        f.seek(header_bytes)
        raw = f.read()
    with np.errstate(invalid="ignore"):
        return np.frombuffer(raw, np.dtype(dtype).newbyteorder("<")).astype(np.float64)


def to_planes(values, shape, layout):
    """(H, W, 2) u, v from the flat values in `layout`."""
    h, w = shape
    if layout == "hw2":
        return values.reshape(h, w, 2)
    return values.reshape(2, h, w).transpose(1, 2, 0)


def flow_factor(shape, frame_shape):
    """k when the flow is the frame divided by one integer k (k = 1: full size).
    Anything else is refused: the band is reduced by whole k x k blocks."""
    (h, w), (fh, fw) = shape, frame_shape
    if fh % h or fw % w or fh // h != fw // w:
        raise KpiError(f"mv shape {h}x{w} is not the frame {fh}x{fw} divided by one "
                       f"integer (1, 2, 4, ...); check --mv-shape")
    return fh // h


def check_unit_res(factor, unit_res):
    """--mv-unit-res quarter is only true of a flow at 1/4 of the frame."""
    if unit_res == "quarter" and factor != MV_UNIT_RES["quarter"]:
        raise KpiError(f"--mv-unit-res quarter, but the flow is 1/{factor} of the "
                       f"frame: give --mv-unit-res full, or the right --mv-shape")


def read_mv(path, fmt, frame_shape):
    """One .dof file as (H, W, 2) float64 u, v in pixels. The one place that
    reads the format: a new layout or a real header goes here."""
    shape = fmt.shape or (frame_shape[0] // MV_DEFAULT_FACTOR,
                          frame_shape[1] // MV_DEFAULT_FACTOR)
    if not fmt.shape and (frame_shape[0] % MV_DEFAULT_FACTOR
                          or frame_shape[1] % MV_DEFAULT_FACTOR):
        raise KpiError(f"{path}: the frame {frame_shape[0]}x{frame_shape[1]} is not "
                       f"divisible by {MV_DEFAULT_FACTOR}, the default flow size; "
                       f"give --mv-shape")
    check_unit_res(flow_factor(shape, frame_shape), fmt.unit_res)
    size = os.path.getsize(path)
    need = fmt.header_bytes + mv_data_bytes(shape, fmt.dtype)
    if size != need:
        raise KpiError("\n".join([
            f"{path}: {size} bytes, but --mv-dtype {fmt.dtype} at {shape[0]}x{shape[1]} "
            f"with --mv-header-bytes {fmt.header_bytes} needs {need} (header + H x W x 2 "
            f"x {np.dtype(fmt.dtype).itemsize}). Sizes that would match "
            f"(--mv-layout does not change the size; run --mv-probe to see the values):",
            *mv_size_table(path, size, frame_shape, fmt.header_bytes)]))
    flow = to_planes(read_mv_values(path, fmt.dtype, fmt.header_bytes),
                     shape, fmt.layout) / fmt.scale
    bad = int((~np.isfinite(flow)).sum())
    if bad:
        raise KpiError(f"{path}: {bad} motion vector value(s) are NaN or inf "
                       f"(as read with --mv-dtype {fmt.dtype}, --mv-scale {fmt.scale:g})")
    return flow


class MvCandidate(NamedTuple):
    dtype: str
    factor: int             # frame / factor
    shape: tuple
    header: int             # the header that makes the size match (may be < 0)


def mv_candidates(size, frame_shape):
    """Every common dtype at the frame size / 1, 2, 4, 8 (when it divides)."""
    fh, fw = frame_shape
    return [MvCandidate(dtype, k, (fh // k, fw // k),
                        size - mv_data_bytes((fh // k, fw // k), dtype))
            for k in MV_RESOLUTIONS if not fh % k and not fw % k
            for dtype in MV_DTYPES]


def mv_fits(candidate, header_bytes):
    """'yes' when it matches with the given header, a header hint, or ''."""
    if candidate.header == header_bytes:
        return "yes"
    if 0 <= candidate.header <= MV_HEADER_HINT:
        return f"with --mv-header-bytes {candidate.header}"
    return ""


def mv_size_table(path, size, frame_shape, header_bytes):
    """The lines that show which flags would fit a file of `size` bytes."""
    lines = [f"  {'dtype':8} {'size':6} {'shape':10} {'data bytes':>10}  fits "
             f"(frame {frame_shape[0]}x{frame_shape[1]}, --mv-header-bytes {header_bytes})"]
    for c in mv_candidates(size, frame_shape):
        lines.append(f"  {c.dtype:8} {'1/' + str(c.factor):6} "
                     f"{f'{c.shape[0]}x{c.shape[1]}':10} {size - c.header:>10}  "
                     f"{mv_fits(c, header_bytes) or '-'}")
    with open(path, "rb") as f:
        if f.read(len(FLO_MAGIC)) == FLO_MAGIC:
            lines.append(f"  note: the file starts with {FLO_MAGIC.decode()}, as a "
                         f"Middlebury .flo does (12-byte header, float32, hw2)")
    return lines


def plane_stats(plane):
    """min / max / median of the finite values, and the mean step between
    horizontal neighbours: small for a real flow, large when interleaved
    (hw2) data is read as planes (u, v, u, ... in one plane)."""
    finite = plane[np.isfinite(plane)]
    if not finite.size:
        return "no finite value"
    step = np.abs(np.diff(plane, axis=1))
    step = step[np.isfinite(step)]
    return (f"min {finite.min():.6g} max {finite.max():.6g} median "
            f"{float(np.median(finite)):.6g} step {step.mean() if step.size else math.nan:.6g}")


def probe_lines(path, frame_shape, header_bytes):
    """The --mv-probe report of one file: the size table, then the values
    (as stored, before --mv-scale) of each candidate whose size fits."""
    size = os.path.getsize(path)
    lines = [f"mv probe: {path}, {size} bytes",
             *mv_size_table(path, size, frame_shape, header_bytes)]
    fitting = [c for c in mv_candidates(size, frame_shape) if mv_fits(c, header_bytes)]
    if not fitting:
        lines.append("no candidate fits: the file is none of these; give "
                     "--mv-shape or --mv-header-bytes")
    for c in fitting:
        values = read_mv_values(path, c.dtype, c.header)
        bad = int((~np.isfinite(values)).sum())
        lines.append(f"{c.dtype} {c.shape[0]}x{c.shape[1]} header {c.header}: "
                     f"{bad} non-finite value(s), as stored (before --mv-scale)")
        for layout in MV_LAYOUTS:
            flow = to_planes(values, c.shape, layout)
            lines.append(f"  {layout}: u {plane_stats(flow[..., 0])} | "
                         f"v {plane_stats(flow[..., 1])}")
    lines.append("probe only: nothing measured, nothing written. The right "
                 "candidate has plausible pixel values and the smaller step.")
    return lines


def band_at(mask, shape):
    """The band at the flow's size: a flow pixel is on it when any frame
    pixel of its k x k block is. Nearest-neighbour sampling would keep one
    pixel per block and drop most of a thin band at 1/4 or 1/8 size."""
    k = flow_factor(shape, mask.shape)
    h, w = shape
    return mask.reshape(h, k, w, k).any(axis=(1, 3))


def mv_sums(band, epe, float_mag):
    """The sums and counts of every region, and the flow's size."""
    h, w = epe.shape
    row = {"mv_H": h, "mv_W": w, "mv_edge_pixels": int(band.sum()), "mv_pixels": h * w}
    for region, sel in (("on_edges", band), ("all", np.ones_like(band))):
        row[f"mv_epe_sum_{region}"] = float(epe[sel].sum())
        row[f"mv_float_mag_sum_{region}"] = float(float_mag[sel].sum())
        for t in MV_EPE_THRESHOLDS:
            row[f"{mv_over(t)[0]}_{region}"] = int((epe[sel] > t).sum())
    return row


def mv_means(row):
    """Each region's means and fractions from its sums and pixel count, in
    place: the same rule for a pair and for the TOTAL."""
    for region, pixels in MV_REGIONS.items():
        n = row[pixels]
        row[f"mv_epe_mean_{region}"] = _ratio(row[f"mv_epe_sum_{region}"], n)
        row[f"mv_float_mag_mean_{region}"] = _ratio(row[f"mv_float_mag_sum_{region}"], n)
        # Unit-free: the same at any --mv-scale and on either grid.
        row[f"mv_rel_epe_{region}"] = _ratio(row[f"mv_epe_sum_{region}"],
                                             row[f"mv_float_mag_sum_{region}"])
        for t in MV_EPE_THRESHOLDS:
            count, frac = mv_over(t)
            row[f"{frac}_{region}"] = _ratio(row[f"{count}_{region}"], n)
    return row


def mv_measures(mask, quant_path, float_path, fmt):
    """The MV_FIELDS of one pair: endpoint error EPE = |quant - float| of
    the (u, v) vectors, per pixel, on the band and on all pixels."""
    quant = read_mv(quant_path, fmt, mask.shape)
    flt = read_mv(float_path, fmt, mask.shape)
    epe = np.hypot(quant[..., 0] - flt[..., 0], quant[..., 1] - flt[..., 1])
    band = band_at(mask, epe.shape)
    return mv_means(mv_sums(band, epe, np.hypot(flt[..., 0], flt[..., 1])))


def mv_total(rows):
    """The mv fields of the TOTAL row: every sum and count summed, the means
    recomputed from them (never averaged)."""
    total = {f: sum(r[f] for r in rows) for f in MV_SUMMED}
    return {"mv_H": "", "mv_W": "", **mv_means(total)}


def with_mv(row, mask, mv_paths, args):
    """The row, with the mv measures when --mv is given."""
    if mv_paths is not None:
        row.update(mv_measures(mask, *mv_paths, mv_format(args)))
    return row


def write_csv(path, rows, fields=CSV_FIELDS):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- CLI

def finite_number(text, expected):
    """float(text) when finite, else argparse's refusal saying `expected`.

    float() also reads 'nan' and 'inf'; an offset or a scale of either turns
    every value into nan or inf, so the KPI would read nan as if measured.
    """
    refused = argparse.ArgumentTypeError(f"expected {expected}, got {text!r}")
    try:
        value = float(text)
    except ValueError:
        raise refused from None
    if not math.isfinite(value):
        raise refused
    return value


def offset_arg(text):
    """An offset flag's value: a finite number, or 'none' for no offset."""
    if text.lower() == "none":
        return None
    return finite_number(text, "a finite number or 'none'")


def scale_arg(text):
    """--mv-scale: a finite number other than 0 (the values are divided by it)."""
    expected = "a finite number other than 0"
    value = finite_number(text, expected)
    if value == 0:
        raise argparse.ArgumentTypeError(f"expected {expected}, got {text!r}")
    return value


def int_at_least(minimum):
    """An argparse type: an integer >= minimum."""
    def parse(text):
        try:
            value = int(text)
        except ValueError:
            value = None
        if value is None or value < minimum:
            raise argparse.ArgumentTypeError(
                f"expected an integer >= {minimum}, got {text!r}")
        return value
    return parse


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images",
                    help="folder of every input frame (both A and B in "
                         "diff-name pairing); edges are found on the "
                         "reference. Y8 inputs (1 channel) preferred (required)")
    # Required, --diffs and --out unless --mv-probe: checked in check_mv_args, so
    # that a missing set is named as argparse names it, all together.
    ap.add_argument("--diffs", help="folder of diff L files (required)")
    ap.add_argument("--out", help="CSV to write (required)")
    ap.add_argument("--allow-unpaired", action="store_true",
                    help="skip (still list) files with no partner instead of failing")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--pair-from-diff-name", dest="from_diff_name",
                      action="store_const", const=True, default=None,
                      help="pair by <A>_<B>_diff_L names (default when every "
                           "diff stem ends in _diff_L)")
    mode.add_argument("--pair-same-stem", dest="from_diff_name",
                      action="store_const", const=False,
                      help="pair files of one stem (default otherwise)")
    ap.add_argument("--ref-frame", choices=("first", "second"), default="first",
                    help="diff-name pairing: which of A, B is the reference "
                         "image the edges are found on (default first: "
                         "DOF infer's im0)")
    ap.add_argument("--diff-channel-tol", type=float, default=DIFF_CHANNEL_TOL,
                    help="a 3-channel diff is gray if its channels differ by "
                         "at most this (default %(default)g, JPEG noise)")
    ap.add_argument("--diff-offset", type=offset_arg, default=None,
                    help="subtracted from png/tif/jpg diff values before abs "
                         "(signed diff stored with an offset, e.g. 128 or "
                         "32768), or 'none'; default none. The run prints the stored "
                         "min/max/median to tell which one a file is")
    ap.add_argument("--diffs-float", default=None,
                    help="folder of the FLOAT run's diff L files, same pairs: "
                         "adds float_*, delta_* (quant - float) and qerr_* "
                         "(|dq - df|) columns; --diffs is the quant run")
    ap.add_argument("--diff-offset-float", type=offset_arg, default=SAME_AS_DIFF_OFFSET,
                    help="--diffs-float's offset, as --diff-offset, or 'none' "
                         "for a plain |diff| float run (default: the value of "
                         "--diff-offset)")
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
    add_mv_args(ap)
    return ap


def add_mv_args(ap):
    """The motion-vector flags. The format flags default to None, so that
    check_args can tell them given; mv_format resolves the defaults."""
    mv = ap.add_argument_group("motion vectors (opt-in; DOF infer's mv/ folders, .dof files)")
    mv.add_argument("--mv", default=None,
                    help="folder of the QUANT run's .dof files, named <A>_<B>.dof "
                         "or <A>_<B>_<suffix>.dof (the diff's pair): adds the "
                         "mv_* columns, endpoint error vs --mv-float")
    mv.add_argument("--mv-float", default=None,
                    help="folder of the FLOAT run's .dof files, same pairs (with --mv)")
    mv.add_argument("--mv-probe", action="store_true",
                    help="read the first --mv file only, print the sizes that fit "
                         "and the values of each; writes nothing (no --diffs/--out)")
    mv.add_argument("--mv-dtype", choices=MV_DTYPES, default=None,
                    help="stored value type, little-endian (default int16)")
    mv.add_argument("--mv-layout", choices=MV_LAYOUTS, default=None,
                    help="2hw: one plane, then the other (default; EPE does not "
                         "depend on which is u); hw2: H x W x (u, v) interleaved")
    mv.add_argument("--mv-header-bytes", type=int_at_least(0), default=None,
                    help="bytes skipped at the start of each file (default 0)")
    mv.add_argument("--mv-scale", type=scale_arg, default=None,
                    help="pixels = stored value / this, e.g. 16 for 4 fraction "
                         "bits of fixed point (default 1: raw units, with a warning)")
    mv.add_argument("--mv-shape", type=int_at_least(1), nargs=2, metavar=("H", "W"),
                    default=None,
                    help="flow size (default: 1/4 of the reference frame); the "
                         "frame divided by an integer k: the band is reduced to it")
    mv.add_argument("--mv-unit-res", choices=tuple(MV_UNIT_RES), default=None,
                    help="the grid the vectors' pixels are on, and so the EPE's: "
                         "quarter (default; the flow must be 1/4 of the frame) or "
                         "full (frame pixels). A label: nothing is rescaled")


MV_FORMAT_FLAGS = ("mv_dtype", "mv_layout", "mv_header_bytes", "mv_scale", "mv_shape",
                   "mv_unit_res")


def mv_format(args):
    """The MvFormat of the --mv-* flags, defaults filled in: DOF infer's
    files as the owner's probe measured them (2026-11-12): int16, planar,
    no header, 1/4 of the frame. Scale and unit are not known: raw, quarter."""
    return MvFormat(args.mv_dtype or "int16", args.mv_layout or "2hw",
                    args.mv_header_bytes or 0,
                    1.0 if args.mv_scale is None else args.mv_scale,
                    args.mv_scale is not None,
                    tuple(args.mv_shape) if args.mv_shape else None,
                    args.mv_unit_res or "quarter")


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
    if args.diff_channel_tol < 0:
        ap.error("--diff-channel-tol must be >= 0")
    if args.diffs_float is None and args.diff_offset_float is not SAME_AS_DIFF_OFFSET:
        ap.error("--diff-offset-float needs --diffs-float")
    check_mv_args(ap, args)


def check_mv_args(ap, args):
    """Every mv flag needs --mv; a run (not the probe) needs --diffs, --out
    and both mv folders."""
    orphans = [f"--{f.replace('_', '-')}" for f in MV_FORMAT_FLAGS if getattr(args, f) is not None]
    orphans += ["--mv-probe"] if args.mv_probe else []
    orphans += ["--mv-float"] if args.mv_float is not None else []
    required = (("--images", args.images),) + (() if args.mv_probe else (
        ("--diffs", args.diffs), ("--out", args.out)))
    missing = [flag for flag, value in required if value is None]
    if missing:                                   # argparse's own words for it
        ap.error(f"the following arguments are required: {', '.join(missing)}")
    if args.mv is None and orphans:
        ap.error(f"{orphans[0]} needs --mv")
    if args.mv_probe:
        return
    if args.mv is not None and args.mv_float is None:
        ap.error("--mv needs --mv-float: the endpoint error compares two runs")


def float_offset(args):
    """--diff-offset-float (None: no offset), or --diff-offset when not given."""
    if args.diff_offset_float is SAME_AS_DIFF_OFFSET:
        return args.diff_offset
    return args.diff_offset_float


def format_measures(row, prefix=""):
    return (f"sum |dL| on edges {row[prefix + 'sum_abs_diff_on_edges']:.6g} | "
            f"mean on edges {row[prefix + 'mean_abs_diff_on_edges']:.6g} | "
            f"sum |dL| all {row[prefix + 'sum_abs_diff_all']:.6g} | "
            f"edge/all {row[prefix + 'ratio_edge_to_all']:.4f}")


def format_total(row):
    return (f"TOTAL | edge_pixels {row['edge_pixels']} "
            f"({row['edge_fraction']:.4%}) | {format_measures(row)}")


def format_compare_total(row):
    """The compare lines printed after format_total."""
    return [f"TOTAL float | {format_measures(row, FLOAT)}",
            f"TOTAL delta quant - float | {format_measures(row, DELTA)}",
            f"TOTAL |dq - df| ({row['diff_kind']}) | {format_measures(row, QERR)}"]


def abs_warning(rows):
    """The warning when a pair holds a plain |diff|, else None."""
    plain = sum(r["diff_kind"] == "abs" for r in rows)
    if not plain:
        return None
    return (f"warning: {plain} of {len(rows)} pair(s) hold a plain |diff| "
            f"file (diff_kind abs; .npy or --diff-offset/--diff-offset-float "
            f"make a diff signed): there |dq - df| is not warped_quant - "
            f"warped_float, only the delta_* columns are meaningful")


class RunPairs(NamedTuple):
    pairs: list             # (name, image, diff), or with --diffs-float (name, image, quant, float)
    notes: list             # note lines for the run summary
    mv: object              # MvFiles with --mv, else None


def pair_runs(args):
    """The RunPairs: every pair of --diffs, or with --diffs-float the pairs
    in both runs, and with --mv only those with both mv files."""
    pairing = pair_folder(_stems(args.images, IMAGE_EXTS, "images"), args.images,
                          args.diffs, args.allow_unpaired, args.from_diff_name,
                          args.ref_frame)
    pairs, notes = pairing.pairs, pairing.notes
    if args.diffs_float is not None:
        # The float run's frames-named-by-no-diff note adds nothing: its frames are
        # the quant run's, or those of pairs in one run only, which match_runs lists.
        float_pairs, _ = pair_files(args.images, args.diffs_float, args.allow_unpaired,
                                    args.from_diff_name, args.ref_frame, "diffs-float")
        pairs, one_run_only = match_runs(pairs, float_pairs, args.allow_unpaired)
        if one_run_only:
            notes.append(listed_note("pair(s) in one run only, skipped in both",
                                     one_run_only))
    if args.mv is None:
        return RunPairs(pairs, notes, None)
    pairs, mv, skipped = match_mv(pairs, pairing, args.mv, args.mv_float,
                                  args.allow_unpaired)
    if skipped:
        notes.append(listed_note("pair(s) without both mv files, skipped", skipped))
    return RunPairs(pairs, notes, mv)


def mv_format_line(fmt, suffixes):
    """The mv flags used, printed on every --mv run like the offset lines."""
    shape = (f"{fmt.shape[0]}x{fmt.shape[1]}" if fmt.shape
             else f"1/{MV_DEFAULT_FACTOR} of the reference frame")
    names = " and ".join(sorted({f"<A>_<B>{s and '_' + s}.dof" for s in suffixes}))
    return (f"mv as read: --mv-dtype {fmt.dtype} (little-endian) | --mv-layout "
            f"{fmt.layout} | --mv-header-bytes {fmt.header_bytes} | --mv-scale "
            f"{fmt.scale:g} | --mv-shape {shape} | --mv-unit-res {fmt.unit_res} | "
            f"files {names}")


def mv_scale_warning(fmt):
    """The warning when --mv-scale is not given, else None."""
    if fmt.scale_given:
        return None
    return ("warning: no --mv-scale: EPE is in the raw stored units (fixed point?), "
            "not pixels, so the > 0.5 px and > 1 px columns are not in pixels "
            "either; mv_rel_epe_* is unit-free. Give --mv-scale (e.g. 2^fraction bits)")


def format_mv_total(row, fmt):
    """The TOTAL lines of the mv measures, one per region."""
    unit = "px" if fmt.scale_given else "raw units"
    lines = []
    for region, pixels in MV_REGIONS.items():
        over = " | ".join(f"> {t:g} {unit} {row[f'{mv_over(t)[1]}_{region}']:.4%}"
                          for t in MV_EPE_THRESHOLDS)
        lines.append(f"TOTAL mv EPE {region} ({fmt.unit_res} grid) | pixels {row[pixels]} | "
                     f"sum {row[f'mv_epe_sum_{region}']:.6g} | mean "
                     f"{row[f'mv_epe_mean_{region}']:.6g} {unit} | {over} | float |flow| "
                     f"mean {row[f'mv_float_mag_mean_{region}']:.6g} {unit} | "
                     f"EPE / |flow| {row[f'mv_rel_epe_{region}']:.4f}")
    return lines


def summary_lines(rows, total, notes, ranges, args, mv=None):
    """The run summary printed to stdout, compare lines only with --diffs-float,
    mv lines only with --mv (mv: the MvFiles)."""
    compare = args.diffs_float is not None
    lines = [f"{len(rows)} pairs -> {args.out}", *notes,
             ranges[0].line(args.diff_offset)]
    if compare:
        lines.append(ranges[1].line(float_offset(args)))
    if mv is not None:
        lines.append(mv_format_line(mv_format(args), mv.suffixes))
    lines.append(format_total(total))
    lines += format_compare_total(total) if compare else []
    return lines + (format_mv_total(total, mv_format(args)) if mv is not None else [])


def probe_frame_shape(stem, args):
    """(H, W) of the reference frame the mv file `stem` names: tried as the
    DOF names <A>_<B>[_<suffix>] unless --pair-same-stem."""
    images = _stems(args.images, IMAGE_EXTS, "images")
    name = None
    for from_diff_name in ((False,) if args.from_diff_name is False else (True, False)):
        name = split_mv_name(stem, images, from_diff_name)
        if name is not None:
            break
    if name is None:
        raise KpiError(f"--mv-probe: {stem} names no frame in --images, so the "
                       f"frame size is not known")
    ref = name.frames[0] if args.ref_frame == "first" else name.frames[-1]
    return _imread(images[ref]).shape[:2]


def run_probe(args):
    """--mv-probe: what the first --mv file can be, printed; nothing written."""
    files = _stems(args.mv, MV_EXTS, "mv")
    if not files:
        raise KpiError(f"--mv: no {'/'.join(MV_EXTS)} file in {args.mv}")
    stem = sorted(files)[0]
    for line in probe_lines(files[stem], probe_frame_shape(stem, args),
                            mv_format(args).header_bytes):
        print(line)


def run(args):
    if cv2 is None:
        raise KpiError("OpenCV is missing: pip install opencv-python-headless")
    if args.mv_probe:
        return run_probe(args)
    compare = args.diffs_float is not None
    pairs, notes, mv = pair_runs(args)
    if args.save_masks:
        os.makedirs(args.save_masks, exist_ok=True)
    ranges = (RawRange(), RawRange("float diff L", "--diff-offset-float"))
    measure = measure_compare if compare else measure_pair
    rows = [measure(*pair, args, ranges, mv.paths[pair[0]] if mv else None)
            for pair in pairs]
    for r in rows:
        if r["edge_pixels"] == 0:
            print(f"note: {r['name']}: no edge pixels, its mean is NaN",
                  file=sys.stderr)
    total = (compare_total_row if compare else total_row)(rows)
    fields = COMPARE_FIELDS if compare else CSV_FIELDS
    if mv:
        total.update(mv_total(rows))
        fields += MV_FIELDS
    write_csv(args.out, rows + [total], fields)
    warnings = [abs_warning(rows) if compare else None,
                mv_scale_warning(mv_format(args)) if mv else None]
    for warning in filter(None, warnings):
        print(warning, file=sys.stderr)
    for line in summary_lines(rows, total, notes, ranges, args, mv):
        print(line)
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
