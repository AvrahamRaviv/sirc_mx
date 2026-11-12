"""tools/dof_edge_kpi.py on synthetic folders; every expected value is built by hand.

The step picture: H rows, W=64 columns, luma 60 left of column 32 and 180 from
it. Canny's non-maximum suppression leaves a one-pixel line on one side of the
step (column 31 or 32), so with --dilate 2 the band is 5 columns wide and
always holds columns 30..33.
"""
import csv
import math
import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2", reason="OpenCV missing: pip install '.[eval]' to run the DOF edge KPI tests")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

import dof_edge_kpi as K

W, STEP = 64, 32
BAND = slice(30, 34)                       # 4 columns inside any 2-px band


def step_image(h=48, dtype=np.uint8):
    img = np.full((h, W), 60, np.uint8)
    img[:, STEP:] = 180
    return img.astype(np.uint16) * 257 if dtype == np.uint16 else img


def band_diff(h=48, value=3.0):
    diff = np.zeros((h, W), np.float32)
    diff[:, BAND] = value
    return diff


def make_dirs(tmp_path):
    images, diffs = tmp_path / "images", tmp_path / "diffs"
    images.mkdir()
    diffs.mkdir()
    return images, diffs


def run_cli(tmp_path, images, diffs, *extra):
    out = tmp_path / "kpi.csv"
    code = K.main(["--images", str(images), "--diffs", str(diffs),
                   "--out", str(out), *extra])
    rows = None
    if out.exists():
        with open(out, newline="") as f:
            rows = {r["name"]: r for r in csv.DictReader(f)}
    return code, rows


def test_diff_on_the_edge_band_gives_the_exact_sum(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff())
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2")
    assert code == 0
    r = rows["a"]
    assert (int(r["H"]), int(r["W"])) == (48, W)
    assert int(r["edge_pixels"]) == 5 * 48          # 1-px line widened by 2 each side
    assert float(r["sum_abs_diff_on_edges"]) == 3.0 * 4 * 48
    assert float(r["sum_abs_diff_all"]) == 3.0 * 4 * 48
    assert float(r["ratio_edge_to_all"]) == 1.0
    assert float(r["mean_abs_diff_on_edges"]) == pytest.approx(576 / 240)
    assert float(r["edge_fraction"]) == pytest.approx(240 / (48 * W))


def test_diff_away_from_edges_gives_edge_sum_zero(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    diff = np.zeros((48, W), np.float32)
    diff[:, :11] = -2.0                             # negative: abs is taken
    diff[:, 50:] = 5.0
    np.save(diffs / "a.npy", diff)
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2")
    assert code == 0
    assert float(rows["a"]["sum_abs_diff_on_edges"]) == 0.0
    assert float(rows["a"]["sum_abs_diff_all"]) == 2 * 11 * 48 + 5 * 14 * 48
    assert float(rows["a"]["ratio_edge_to_all"]) == 0.0


def test_flat_image_has_no_edges_mean_nan_and_counts_in_total(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "flat.png"), np.full((48, W), 90, np.uint8))
    np.save(diffs / "flat.npy", np.ones((48, W), np.float32))
    cv2.imwrite(str(images / "step.png"), step_image())
    np.save(diffs / "step.npy", band_diff())
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2")
    assert code == 0
    flat = rows["flat"]
    assert int(flat["edge_pixels"]) == 0
    assert math.isnan(float(flat["mean_abs_diff_on_edges"]))
    assert float(flat["sum_abs_diff_all"]) == 48 * W
    total = rows["TOTAL"]
    assert int(total["edge_pixels"]) == 240
    assert float(total["edge_fraction"]) == pytest.approx(240 / (2 * 48 * W))
    assert float(total["sum_abs_diff_all"]) == 48 * W + 576
    assert "flat: no edge pixels" in capsys.readouterr().err


def test_total_recomputes_means_from_sums_never_averages_them(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "big.png"), step_image(48))
    np.save(diffs / "big.npy", band_diff(48, 3.0))  # 576 over 240 edge px
    cv2.imwrite(str(images / "small.png"), step_image(24))
    np.save(diffs / "small.npy", band_diff(24, 1.0))  # 96 over 120 edge px
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2")
    assert code == 0
    total = rows["TOTAL"]
    assert int(total["edge_pixels"]) == 360
    assert float(total["sum_abs_diff_on_edges"]) == 672
    assert float(total["mean_abs_diff_on_edges"]) == pytest.approx(672 / 360)
    assert total["H"] == "" and total["W"] == ""


@pytest.mark.parametrize("detector", ["canny", "sobel"])
def test_16_bit_and_8_bit_give_the_same_mask(tmp_path, detector):
    p8, p16 = tmp_path / "a8.png", tmp_path / "a16.png"
    rng = np.random.default_rng(0)
    pic = step_image()
    pic[10:20, 5:25] = rng.integers(0, 256, (10, 20), dtype=np.uint8)
    cv2.imwrite(str(p8), pic)
    cv2.imwrite(str(p16), pic.astype(np.uint16) * 257)
    assert K.load_luma(str(p16)).dtype == np.float32
    args = K.build_parser().parse_args(
        ["--images", "x", "--diffs", "y", "--out", "z", "--detector", detector])
    m8 = K.edge_mask(K.load_luma(str(p8)), args)
    m16 = K.edge_mask(K.load_luma(str(p16)), args)
    assert m8.any()
    assert np.array_equal(m8, m16)


def test_color_is_reduced_with_bt601(tmp_path):
    for bgr, weight in (((0, 0, 255), 0.299), ((0, 255, 0), 0.587), ((255, 0, 0), 0.114)):
        path = tmp_path / "c.png"
        cv2.imwrite(str(path), np.full((4, 4, 3), bgr, np.uint8))
        assert K.load_luma(str(path)) == pytest.approx(np.full((4, 4), weight), abs=1e-6)


@pytest.mark.parametrize("dtype, offset", [(np.uint8, 128), (np.uint16, 32768)])
def test_signed_diff_with_offset_takes_abs_after_offset(tmp_path, dtype, offset, capsys):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    stored = np.full((48, W), offset, dtype)
    stored[:, 30:32] = offset - 5                   # diff -5 on the band
    stored[:, 32:34] = offset + 3                   # diff +3 on the band
    stored[:, :4] = offset - 7                      # diff -7 away from edges
    cv2.imwrite(str(diffs / "a.png"), stored)
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2",
                         "--diff-offset", str(offset))
    assert code == 0
    assert float(rows["a"]["sum_abs_diff_on_edges"]) == (5 * 2 + 3 * 2) * 48
    assert float(rows["a"]["sum_abs_diff_all"]) == (5 * 2 + 3 * 2 + 7 * 4) * 48
    out = capsys.readouterr().out
    assert (f"min {offset - 7}, max {offset + 3}, median of per-file medians "
            f"{offset} | --diff-offset {offset}") in out


def test_offset_never_touches_a_npy_diff(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff())           # already signed: 3 on the band
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2", "--diff-offset", "128")
    assert code == 0
    assert float(rows["a"]["sum_abs_diff_all"]) == 3.0 * 4 * 48


def test_pairs_by_stem_across_extensions(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "img_001.png"), step_image())
    np.save(diffs / "img_001.npy", band_diff())
    cv2.imwrite(str(images / "img_002.tif"), step_image())
    cv2.imwrite(str(diffs / "img_002.png"), np.zeros((48, W), np.uint8))
    pairs, notes = K.pair_files(str(images), str(diffs))
    assert notes == []
    assert [(s, os.path.basename(i), os.path.basename(d)) for s, i, d in pairs] == [
        ("img_001", "img_001.png", "img_001.npy"),
        ("img_002", "img_002.tif", "img_002.png")]


def test_unpaired_files_are_listed_and_an_error_unless_allowed(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff())
    cv2.imwrite(str(images / "only_image.png"), step_image())
    np.save(diffs / "only_diff.npy", band_diff())
    code, rows = run_cli(tmp_path, images, diffs)
    err = capsys.readouterr().err
    assert code == 2 and rows is None
    assert "image without a diff: only_image" in err
    assert "diff without an image: only_diff" in err
    code, rows = run_cli(tmp_path, images, diffs, "--allow-unpaired")
    err = capsys.readouterr().err
    assert code == 0 and set(rows) == {"a", "TOTAL"}
    assert "image without a diff: only_image" in err


def test_two_files_with_one_stem_is_an_error(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    cv2.imwrite(str(images / "a.tif"), step_image())
    np.save(diffs / "a.npy", band_diff())
    code, _ = run_cli(tmp_path, images, diffs)
    assert code == 2
    assert "two files for 'a'" in capsys.readouterr().err


def test_shape_mismatch_is_an_error_naming_the_file(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", np.zeros((40, W), np.float32))
    code, rows = run_cli(tmp_path, images, diffs)
    assert code == 2 and rows is None
    err = capsys.readouterr().err
    assert "a.npy" in err and "48x64" in err and "40x64" in err


def test_empty_folder_is_an_error_not_a_csv_of_zeros(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    code, rows = run_cli(tmp_path, images, diffs)
    assert code == 2 and rows is None
    assert "no (image, diff) pairs" in capsys.readouterr().err


@pytest.mark.parametrize("bad, message", [
    (np.full((48, W), np.nan, np.float32), "NaN or inf"),
    (np.zeros((2, 48, W), np.float32), "expected (H, W)"),
])
def test_bad_diff_is_an_error(tmp_path, capsys, bad, message):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", bad)
    code, _ = run_cli(tmp_path, images, diffs)
    assert code == 2
    assert message in capsys.readouterr().err


def test_sobel_cross_check_finds_the_same_step(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff())
    masks = tmp_path / "masks"
    code, rows = run_cli(tmp_path, images, diffs, "--detector", "sobel",
                         "--sigma", "0", "--dilate", "0", "--save-masks", str(masks))
    assert code == 0
    mask = cv2.imread(str(masks / "a.png"), cv2.IMREAD_UNCHANGED)
    cols = sorted(set(np.where(mask > 0)[1]))
    # 3x3 Sobel on a step answers on the two columns beside it, nowhere else.
    assert cols == [STEP - 1, STEP]
    assert int(rows["a"]["edge_pixels"]) == 2 * 48
    assert float(rows["a"]["sum_abs_diff_on_edges"]) == 3.0 * 2 * 48


def test_saved_mask_is_the_counted_mask(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff())
    masks = tmp_path / "masks"
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2", "--save-masks", str(masks))
    assert code == 0
    mask = cv2.imread(str(masks / "a.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(mask)) == {0, 255}
    assert int((mask > 0).sum()) == int(rows["a"]["edge_pixels"]) == 240


def test_fixed_canny_threshold_above_the_step_gradient_finds_nothing():
    # A 120-level step gives an L2 Sobel magnitude of 4 * 120 = 480 at most.
    luma = step_image().astype(np.float32) / 255
    assert not K.canny_mask(luma, 0, low=600, high=1000).any()
    assert K.canny_mask(luma, 0, low=100, high=200).any()


def test_canny_low_without_high_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        K.main(["--images", str(tmp_path), "--diffs", str(tmp_path),
                "--out", str(tmp_path / "o.csv"), "--canny-low", "10"])


# ---------------------------------------------- the DOF tool's files
# Names of the real shape, synthetic content: <A>_<B>_diff_L.<ext>, A and B
# image stems that hold underscores themselves; A (im0) is the reference.
A, B = "0115_IMX874_Front_142", "0115_IMX874_Front_141"
AB = f"{A}_{B}_diff_L"


def diff_name_folder(tmp_path, diff_ext=".npy"):
    """A = step picture, B = flat picture; the diff has 3 on the step band."""
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / f"{A}.png"), step_image())
    cv2.imwrite(str(images / f"{B}.png"), np.full((48, W), 90, np.uint8))
    if diff_ext == ".npy":
        np.save(diffs / f"{AB}.npy", band_diff())
    else:
        cv2.imwrite(str(diffs / f"{AB}{diff_ext}"), band_diff().astype(np.uint8))
    return images, diffs


@pytest.mark.parametrize("ref_frame, edge_pixels, on_edges", [
    ("first", 240, 576.0),     # A, the step: the 5-column band
    ("second", 0, 0.0),        # B, flat: no edges
])
def test_diff_name_pairs_with_the_reference_frame(tmp_path, capsys, ref_frame,
                                                  edge_pixels, on_edges):
    images, diffs = diff_name_folder(tmp_path)
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2", "--ref-frame", ref_frame)
    assert code == 0
    assert set(rows) == {AB, "TOTAL"}
    assert int(rows[AB]["edge_pixels"]) == edge_pixels
    assert float(rows[AB]["sum_abs_diff_on_edges"]) == on_edges
    assert float(rows[AB]["sum_abs_diff_all"]) == 576.0
    assert f"edges on the {ref_frame} frame" in capsys.readouterr().out


def test_diff_name_default_reference_is_first_and_said(tmp_path, capsys):
    images, diffs = diff_name_folder(tmp_path)
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2")
    assert code == 0
    assert int(rows[AB]["edge_pixels"]) == 240             # A, the step frame
    assert float(rows[AB]["sum_abs_diff_on_edges"]) == 576.0
    assert "edges on the first frame (A, --ref-frame first)" in capsys.readouterr().out


def test_frames_no_diff_names_are_a_note_not_an_error(tmp_path, capsys):
    # A dataset folder: frames 141..149, only the 142_141 diff. 143..149 (7
    # frames) are in no diff: a note with the count and the first 5, exit 0.
    images, diffs = diff_name_folder(tmp_path)
    lone = [f"0115_IMX874_Front_{f}" for f in range(143, 150)]
    for stem in lone:
        cv2.imwrite(str(images / f"{stem}.png"), step_image())
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2")
    out = capsys.readouterr().out
    assert code == 0 and set(rows) == {AB, "TOTAL"}
    assert int(rows["TOTAL"]["edge_pixels"]) == 240            # 142 only
    assert (f"note: 7 frame(s) in --images named by no diff, not measured: "
            f"{', '.join(lone[:5])}, ...") in out
    summary = out[out.index("1 pairs ->"):]                    # the run summary
    assert "note: 7 frame(s)" in summary and "TOTAL |" in summary


def test_bmp_frames_and_y8_png_frames_pair_by_their_real_names(tmp_path):
    # DOF infer's two name shapes: bmp dataset frames, and Y8 PNG inputs.
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / f"{A}.bmp"), step_image())
    cv2.imwrite(str(images / f"{B}.bmp"), step_image())
    cv2.imwrite(str(diffs / f"{AB}.jpg"), np.zeros((48, W, 3), np.uint8))
    y8_a, y8_b = "0015_GN3_001", "0015_GN3_000"
    cv2.imwrite(str(images / f"{y8_a}.png"), step_image(24))
    cv2.imwrite(str(images / f"{y8_b}.png"), np.full((24, W), 90, np.uint8))
    np.save(diffs / f"{y8_a}_{y8_b}_diff_L.npy", band_diff(24, 1.0))
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2")
    assert code == 0
    assert set(rows) == {AB, f"{y8_a}_{y8_b}_diff_L", "TOTAL"}
    assert int(rows[AB]["edge_pixels"]) == 240                       # bmp step read
    y8 = rows[f"{y8_a}_{y8_b}_diff_L"]
    assert int(y8["edge_pixels"]) == 5 * 24                          # edges on A (step)
    assert float(y8["sum_abs_diff_on_edges"]) == 1.0 * 4 * 24


def test_one_channel_image_is_read_as_is(tmp_path):
    path = tmp_path / "y8.png"
    y8 = (np.arange(48 * W) % 256).astype(np.uint8).reshape(48, W)
    cv2.imwrite(str(path), y8)
    luma = K.load_luma(str(path))
    assert luma.shape == (48, W)
    assert np.array_equal(luma, (y8 / 255.0).astype(np.float32))      # no weights


def test_jpg_diff_with_equal_channels_is_read_as_gray(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / f"{A}.png"), step_image())
    cv2.imwrite(str(images / f"{B}.png"), step_image())
    # A constant JPEG decodes exactly, so the sum is known: 7 per pixel.
    cv2.imwrite(str(diffs / f"{AB}.jpg"), np.full((48, W, 3), 7, np.uint8))
    raw = K.read_diff(str(diffs / f"{AB}.jpg")).values
    assert raw.shape == (48, W)
    code, rows = run_cli(tmp_path, images, diffs)
    assert code == 0
    assert float(rows[AB]["sum_abs_diff_all"]) == 7 * 48 * W


def test_default_channel_tolerance_accepts_a_spread_of_2(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    diff = np.full((48, W, 3), 10, np.uint8)
    diff[..., 1] = 12                                       # spread exactly 2
    cv2.imwrite(str(diffs / "a.png"), diff)
    code, rows = run_cli(tmp_path, images, diffs)           # no --diff-channel-tol
    assert code == 0
    assert float(rows["a"]["sum_abs_diff_all"]) == 10 * 48 * W   # channel 0


def test_diff_of_one_row_or_column_is_kept(tmp_path):
    for shape in ((1, W), (48, 1), (1, W, 1)):
        path = tmp_path / "d.npy"
        np.save(path, np.ones(shape))
        assert K.read_diff(str(path)).values.shape == shape[:2]


def test_3_channel_diff_within_tolerance_takes_channel_0(tmp_path):
    path = tmp_path / "d.png"
    diff = np.full((8, 8, 3), 10, np.uint8)
    diff[..., 1] = 12                                       # +2: JPEG-like noise
    diff[..., 2] = 9
    cv2.imwrite(str(path), diff)
    raw = K.read_diff(str(path), channel_tol=3).values
    assert np.array_equal(raw, np.full((8, 8), 10.0))       # channel 0 (B), not luma


def test_3_channel_diff_beyond_tolerance_is_an_error_naming_the_spread(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    diff = np.zeros((48, W, 3), np.uint8)
    diff[..., 2] = 3                                        # a color diff, spread 3
    cv2.imwrite(str(diffs / "a.png"), diff)
    code, _ = run_cli(tmp_path, images, diffs)
    err = capsys.readouterr().err
    assert code == 2
    assert "a.png" in err and "differ by up to 3" in err
    code, rows = run_cli(tmp_path, images, diffs, "--diff-channel-tol", "3")
    assert code == 0 and float(rows["a"]["sum_abs_diff_all"]) == 0.0


def test_ambiguous_diff_name_split_is_an_error(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    for stem in ("cam_1", "cam_1_x", "x_cam_2", "cam_2"):
        cv2.imwrite(str(images / f"{stem}.png"), step_image())
    np.save(diffs / "cam_1_x_cam_2_diff_L.npy", band_diff())   # cam_1|x_cam_2 or cam_1_x|cam_2
    code, rows = run_cli(tmp_path, images, diffs)
    err = capsys.readouterr().err
    assert code == 2 and rows is None
    assert "cam_1_x_cam_2_diff_L" in err and "ambiguous" in err


def test_diff_name_with_no_matching_images_is_listed_and_an_error(tmp_path, capsys):
    images, diffs = diff_name_folder(tmp_path)
    lost = f"{A}_0115_IMX874_Front_999_diff_L"
    np.save(diffs / f"{lost}.npy", band_diff())
    code, rows = run_cli(tmp_path, images, diffs)
    err = capsys.readouterr().err
    assert code == 2 and rows is None
    assert f"splits into no two images: {lost}" in err
    code, rows = run_cli(tmp_path, images, diffs, "--allow-unpaired")
    assert code == 0 and set(rows) == {AB, "TOTAL"}


def test_mixed_diff_names_are_an_error_unless_a_mode_is_forced(tmp_path, capsys):
    images, diffs = diff_name_folder(tmp_path)
    np.save(diffs / f"{A}.npy", band_diff())                   # a same-stem diff too
    code, _ = run_cli(tmp_path, images, diffs)
    assert code == 2 and "mixes *_diff_L names" in capsys.readouterr().err
    code, rows = run_cli(tmp_path, images, diffs, "--pair-same-stem", "--allow-unpaired")
    assert code == 0 and set(rows) == {A, "TOTAL"}


def test_forced_diff_name_pairing_lists_another_name_and_skips_it_if_allowed(tmp_path, capsys):
    images, diffs = diff_name_folder(tmp_path)
    np.save(diffs / "a_1_a_0_flow.npy", band_diff())              # not a diff L name
    code, rows = run_cli(tmp_path, images, diffs, "--pair-from-diff-name")
    assert code == 2 and rows is None
    assert "diff not named <A>_<B>_diff_L: a_1_a_0_flow" in capsys.readouterr().err
    code, rows = run_cli(tmp_path, images, diffs, "--pair-from-diff-name", "--allow-unpaired")
    assert code == 0 and set(rows) == {AB, "TOTAL"}
    assert "diff not named <A>_<B>_diff_L: a_1_a_0_flow" in capsys.readouterr().err


def test_default_canny_thresholds_are_0_66_and_1_33_of_the_median():
    assert K.canny_thresholds(np.full((4, 4), 100, np.uint8)) == pytest.approx((66.0, 133.0))
    lo, hi = K.canny_thresholds(np.full((4, 4), 200, np.uint8))
    assert lo == pytest.approx(132.0) and hi == 255.0              # 266 clipped to 255


@pytest.mark.parametrize("extra, cols", [
    ((), [STEP - 1, STEP, 47, 48]),                    # default 90: both steps
    (("--sobel-percentile", "95"), [STEP - 1, STEP]),  # 95: the strong step only
])
def test_sobel_percentile_keeps_the_strongest_gradients(tmp_path, extra, cols):
    # Strong step 60 -> 180 at column 32, weak step of 10 levels at column 48.
    # A 3x3 Sobel answers on the two columns beside each step: 4 of 64 columns
    # are non-zero, 2 strong + 2 weak. The 90th percentile is 0 (60 zero
    # columns), so all 4 pass; the 95th falls among the weak ones, so only the
    # strong 2 are above it.
    images, diffs = make_dirs(tmp_path)
    img = step_image()
    img[:, 48:] = 190
    cv2.imwrite(str(images / "a.png"), img)
    np.save(diffs / "a.npy", band_diff())
    masks = tmp_path / "masks"
    code, _ = run_cli(tmp_path, images, diffs, "--detector", "sobel", "--sigma", "0",
                      "--dilate", "0", "--save-masks", str(masks), *extra)
    assert code == 0
    mask = cv2.imread(str(masks / "a.png"), cv2.IMREAD_UNCHANGED)
    assert sorted(set(np.where(mask > 0)[1])) == cols


def test_canny_uses_the_l2_gradient():
    # A 45-degree step: L1 |gx|+|gy| is sqrt(2) times L2 there. With a high
    # threshold between the two magnitudes only the L1 norm would pass it.
    luma = (np.add.outer(np.arange(48), np.arange(W)) >= 48).astype(np.float32) * (20 / 255)
    luma8 = np.rint(luma * 255).astype(np.uint8)
    gx = cv2.Sobel(luma8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma8, cv2.CV_32F, 0, 1, ksize=3)
    l2, l1 = float(np.hypot(gx, gy).max()), float((np.abs(gx) + np.abs(gy)).max())
    assert l2 < l1
    high = (l2 + l1) / 2
    assert not K.canny_mask(luma, 0, low=high, high=high).any()


# ---------------------------------------------- no --diffs-float / --mv: unchanged
# The output of e026830 (the tool before --diffs-float) on the first two
# folders, and of e008706 (the tool before --mv) on "compare", run and pasted
# here: CSV bytes, stdout (the CSV path as <OUT>), stderr.
# The --diffs-float CSV header of e008706, written out (not COMPARE_HEADER, so
# that a change there cannot move the pin with it).
COMPARE_HEADER_E008706 = (
    "name", "H", "W", "edge_pixels", "edge_fraction", "sum_abs_diff_on_edges",
    "mean_abs_diff_on_edges", "sum_abs_diff_all", "ratio_edge_to_all",
    "float_sum_abs_diff_on_edges", "float_mean_abs_diff_on_edges", "float_sum_abs_diff_all",
    "float_ratio_edge_to_all", "delta_sum_abs_diff_on_edges", "delta_mean_abs_diff_on_edges",
    "delta_sum_abs_diff_all", "delta_ratio_edge_to_all", "qerr_sum_abs_diff_on_edges",
    "qerr_mean_abs_diff_on_edges", "qerr_sum_abs_diff_all", "qerr_ratio_edge_to_all",
    "diff_kind")
PIN_HEADER = ("name,H,W,edge_pixels,edge_fraction,sum_abs_diff_on_edges,"
              "mean_abs_diff_on_edges,sum_abs_diff_all,ratio_edge_to_all\r\n")
PIN = {
    "diff_name": (
        PIN_HEADER
        + f"{B}_{A}_diff_L,48,64,0,0.0,0.0,nan,21504.0,0.0\r\n"
        + f"{AB},48,64,240,0.078125,576.0,2.4,576.0,1.0\r\n"
        + "TOTAL,,,240,0.0390625,576.0,2.4,22080.0,0.02608695652173913\r\n",
        "pairing: from <A>_<B>_diff_L names, edges on the first frame (A, --ref-frame first)\n"
        "2 pairs -> <OUT>\n"
        "note: 1 frame(s) in --images named by no diff, not measured: 0115_IMX874_Front_143\n"
        "diff L as stored: min 0, max 7, median of per-file medians 3.5 | --diff-offset none\n"
        "TOTAL | edge_pixels 240 (3.9062%) | sum |dL| on edges 576 | mean on edges 2.4 "
        "| sum |dL| all 22080 | edge/all 0.0261\n",
        f"note: {B}_{A}_diff_L: no edge pixels, its mean is NaN\n"),
    "same_stem_offset": (
        PIN_HEADER
        + "a,48,64,384,0.125,768.0,2.0,2112.0,0.36363636363636365\r\n"
        + "b,24,64,192,0.125,144.0,0.75,144.0,1.0\r\n"
        + "TOTAL,,,576,0.125,912.0,1.5833333333333333,2256.0,0.40425531914893614\r\n",
        "2 pairs -> <OUT>\n"
        "diff L as stored: min -1.5, max 131, median of per-file medians 64 | --diff-offset 128\n"
        "TOTAL | edge_pixels 576 (12.5000%) | sum |dL| on edges 912 | mean on edges 1.58333 "
        "| sum |dL| all 2256 | edge/all 0.4043\n",
        ""),
    "compare": (
        ",".join(COMPARE_HEADER_E008706) + "\r\n"
        + "a,48,64,240,0.078125,576.0,2.4,576.0,1.0,192.0,0.8,192.0,1.0,384.0,"
          "1.5999999999999999,384.0,0.0,768.0,3.2,768.0,1.0,signed\r\n"
        + "b,24,64,120,0.078125,288.0,2.4,432.0,0.6666666666666666,96.0,0.8,96.0,1.0,"
          "192.0,1.5999999999999999,336.0,-0.33333333333333337,192.0,1.6,336.0,"
          "0.5714285714285714,signed\r\n"
        + "TOTAL,,,360,0.078125,864.0,2.4,1008.0,0.8571428571428571,288.0,0.8,288.0,1.0,"
          "576.0,1.5999999999999999,720.0,-0.1428571428571429,960.0,2.6666666666666665,"
          "1104.0,0.8695652173913043,signed\r\n",
        "2 pairs -> <OUT>\n"
        "diff L as stored: min 0, max 131, median of per-file medians 64 | --diff-offset 128\n"
        "float diff L as stored: min -1, max 129, median of per-file medians 64 "
        "| --diff-offset-float 128\n"
        "TOTAL | edge_pixels 360 (7.8125%) | sum |dL| on edges 864 | mean on edges 2.4 "
        "| sum |dL| all 1008 | edge/all 0.8571\n"
        "TOTAL float | sum |dL| on edges 288 | mean on edges 0.8 | sum |dL| all 288 "
        "| edge/all 1.0000\n"
        "TOTAL delta quant - float | sum |dL| on edges 576 | mean on edges 1.6 "
        "| sum |dL| all 720 | edge/all -0.1429\n"
        "TOTAL |dq - df| (signed) | sum |dL| on edges 960 | mean on edges 2.66667 "
        "| sum |dL| all 1104 | edge/all 0.8696\n",
        ""),
}


def pin_folder(tmp_path, scenario):
    images, diffs = make_dirs(tmp_path)
    if scenario == "diff_name":
        cv2.imwrite(str(images / f"{A}.png"), step_image())
        cv2.imwrite(str(images / f"{B}.png"), np.full((48, W), 90, np.uint8))
        cv2.imwrite(str(images / "0115_IMX874_Front_143.png"), step_image())
        np.save(diffs / f"{AB}.npy", band_diff())
        cv2.imwrite(str(diffs / f"{B}_{A}_diff_L.png"), np.full((48, W, 3), 7, np.uint8))
        return images, diffs, ["--dilate", "2"]
    if scenario == "compare":
        floats = tmp_path / "floats"
        floats.mkdir()
        cv2.imwrite(str(images / "a.png"), step_image())
        cv2.imwrite(str(images / "b.png"), step_image(24))
        np.save(diffs / "a.npy", band_diff(48, 3.0))
        np.save(floats / "a.npy", band_diff(48, -1.0))
        stored = np.full((24, W), 128, np.uint8)
        stored[:, 30:34], stored[:, :3] = 131, 126
        cv2.imwrite(str(diffs / "b.png"), stored)
        stored = np.full((24, W), 128, np.uint8)
        stored[:, 30:34] = 129
        cv2.imwrite(str(floats / "b.png"), stored)
        return images, diffs, ["--diffs-float", str(floats), "--dilate", "2",
                               "--diff-offset", "128"]
    cv2.imwrite(str(images / "a.png"), step_image())
    cv2.imwrite(str(images / "b.png"), step_image(24))
    stored = np.full((48, W), 128, np.uint8)
    stored[:, 30:32], stored[:, 32:34], stored[:, :4] = 123, 131, 121
    cv2.imwrite(str(diffs / "a.png"), stored)
    np.save(diffs / "b.npy", band_diff(24, -1.5))
    return images, diffs, ["--detector", "sobel", "--diff-offset", "128"]


@pytest.mark.parametrize("scenario", sorted(PIN))
def test_without_the_new_flags_the_output_is_byte_identical(tmp_path, capsys, scenario):
    images, diffs, extra = pin_folder(tmp_path, scenario)
    capsys.readouterr()
    code, _ = run_cli(tmp_path, images, diffs, *extra)
    out = capsys.readouterr()
    csv_text, stdout, stderr = PIN[scenario]
    assert code == 0
    assert (tmp_path / "kpi.csv").read_bytes() == csv_text.encode()
    assert out.out.replace(str(tmp_path / "kpi.csv"), "<OUT>") == stdout
    assert out.err == stderr


# ---------------------------------------------- --diffs-float: float vs quant
COMPARE_HEADER = [
    "name", "H", "W", "edge_pixels", "edge_fraction",
    "sum_abs_diff_on_edges", "mean_abs_diff_on_edges", "sum_abs_diff_all", "ratio_edge_to_all",
    "float_sum_abs_diff_on_edges", "float_mean_abs_diff_on_edges",
    "float_sum_abs_diff_all", "float_ratio_edge_to_all",
    "delta_sum_abs_diff_on_edges", "delta_mean_abs_diff_on_edges",
    "delta_sum_abs_diff_all", "delta_ratio_edge_to_all",
    "qerr_sum_abs_diff_on_edges", "qerr_mean_abs_diff_on_edges",
    "qerr_sum_abs_diff_all", "qerr_ratio_edge_to_all",
    "diff_kind"]


def compare_dirs(tmp_path):
    images, diffs = make_dirs(tmp_path)
    floats = tmp_path / "floats"
    floats.mkdir()
    return images, diffs, floats


def run_compare(tmp_path, images, diffs, floats, *extra):
    return run_cli(tmp_path, images, diffs, "--diffs-float", str(floats), *extra)


def f(row, field):
    return float(row[field])


def test_compare_csv_has_the_compare_columns_in_order(tmp_path):
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff())
    np.save(floats / "a.npy", band_diff())
    code, _ = run_compare(tmp_path, images, diffs, floats, "--dilate", "2")
    assert code == 0
    with open(tmp_path / "kpi.csv", newline="") as fh:
        assert next(csv.reader(fh)) == COMPARE_HEADER


def test_compare_cancels_the_float_error_the_two_runs_share(tmp_path):
    # Band (240 px, diff on its 4 inner columns): quant +3, float +1. Away from
    # edges both runs -2 on columns 0..10: the float model's own error.
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    quant, flt = band_diff(value=3.0), band_diff(value=1.0)
    quant[:, :11] = flt[:, :11] = -2.0
    np.save(diffs / "a.npy", quant)
    np.save(floats / "a.npy", flt)
    code, rows = run_compare(tmp_path, images, diffs, floats, "--dilate", "2")
    assert code == 0
    r = rows["a"]
    assert int(r["edge_pixels"]) == 240
    assert f(r, "sum_abs_diff_on_edges") == 3 * 4 * 48                  # 576
    assert f(r, "sum_abs_diff_all") == 576 + 2 * 11 * 48                # 1632
    assert f(r, "float_sum_abs_diff_on_edges") == 1 * 4 * 48            # 192
    assert f(r, "float_sum_abs_diff_all") == 192 + 2 * 11 * 48          # 1248
    assert f(r, "float_mean_abs_diff_on_edges") == pytest.approx(192 / 240)
    assert f(r, "delta_sum_abs_diff_on_edges") == 576 - 192
    assert f(r, "delta_mean_abs_diff_on_edges") == pytest.approx((576 - 192) / 240)
    assert f(r, "delta_sum_abs_diff_all") == 1632 - 1248
    assert f(r, "qerr_sum_abs_diff_on_edges") == 2 * 4 * 48             # |3 - 1|
    assert f(r, "qerr_sum_abs_diff_all") == 2 * 4 * 48                  # -2 cancels
    assert f(r, "qerr_ratio_edge_to_all") == 1.0
    assert r["diff_kind"] == "signed"


def test_qerr_sees_a_sign_flip_the_delta_cannot(tmp_path):
    # Same size, opposite sign: |dq| and |df| agree, dq - df = 6 on the band.
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff(value=3.0))
    np.save(floats / "a.npy", band_diff(value=-3.0))
    code, rows = run_compare(tmp_path, images, diffs, floats, "--dilate", "2")
    assert code == 0
    assert f(rows["a"], "delta_sum_abs_diff_on_edges") == 0.0
    assert f(rows["a"], "qerr_sum_abs_diff_on_edges") == 6 * 4 * 48


def test_compare_total_is_built_from_summed_counts(tmp_path):
    # big: 240 edge px, quant 3 float 1; small: 120 edge px, quant 1 float 0.
    # qerr sums 384 + 96 = 480 over 360 px: mean 4/3, not the average of the
    # two means (1.6 + 0.8) / 2 = 1.2.
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "big.png"), step_image(48))
    cv2.imwrite(str(images / "small.png"), step_image(24))
    np.save(diffs / "big.npy", band_diff(48, 3.0))
    np.save(floats / "big.npy", band_diff(48, 1.0))
    np.save(diffs / "small.npy", band_diff(24, 1.0))
    np.save(floats / "small.npy", band_diff(24, 0.0))
    code, rows = run_compare(tmp_path, images, diffs, floats, "--dilate", "2")
    assert code == 0
    t = rows["TOTAL"]
    assert int(t["edge_pixels"]) == 360
    assert f(t, "sum_abs_diff_on_edges") == 576 + 96
    assert f(t, "float_sum_abs_diff_on_edges") == 192 + 0
    assert f(t, "float_mean_abs_diff_on_edges") == pytest.approx(192 / 360)
    assert f(t, "delta_sum_abs_diff_on_edges") == 672 - 192
    assert f(t, "delta_mean_abs_diff_on_edges") == pytest.approx((672 - 192) / 360)
    assert f(t, "qerr_sum_abs_diff_on_edges") == 384 + 96
    assert f(t, "qerr_mean_abs_diff_on_edges") == pytest.approx(480 / 360)
    assert f(t, "qerr_sum_abs_diff_all") == 480
    assert t["diff_kind"] == "signed"


def offset_png(path, offset, band_value, dtype=np.uint8):
    stored = np.full((48, W), offset, dtype)
    stored[:, BAND] = offset + band_value
    cv2.imwrite(str(path), stored)


@pytest.mark.parametrize("float_file, extra, float_offset_shown", [
    ("npy", (), "128"),                                  # npy: never offset
    ("png128", (), "128"),                               # default: --diff-offset
    ("png100", ("--diff-offset-float", "100"), "100"),   # its own offset
])
def test_each_run_takes_its_own_offset_and_format(tmp_path, capsys, float_file, extra,
                                                  float_offset_shown):
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    offset_png(diffs / "a.png", 128, 3)                  # quant: +3 on the band
    if float_file == "npy":
        np.save(floats / "a.npy", band_diff(value=1.0))
    else:
        offset_png(floats / "a.png", int(float_file[3:]), 1)   # float: +1
    code, rows = run_compare(tmp_path, images, diffs, floats, "--dilate", "2",
                             "--diff-offset", "128", *extra)
    assert code == 0
    r = rows["a"]
    assert f(r, "sum_abs_diff_all") == 3 * 4 * 48
    assert f(r, "float_sum_abs_diff_all") == 1 * 4 * 48
    assert f(r, "qerr_sum_abs_diff_all") == 2 * 4 * 48
    assert r["diff_kind"] == "signed"
    out = capsys.readouterr().out
    assert f"| --diff-offset-float {float_offset_shown}\n" in out
    assert "\nfloat diff L as stored: " in out


def test_plain_abs_diffs_are_compared_with_a_warning_and_said_in_the_csv(tmp_path, capsys):
    # The DOF tool's own files: plain |diff| png, no offset. a: both plain;
    # b: both .npy (signed). Only delta is the meaningful column for a.
    images, diffs, floats = compare_dirs(tmp_path)
    for stem in ("a", "b"):
        cv2.imwrite(str(images / f"{stem}.png"), step_image())
    cv2.imwrite(str(diffs / "a.png"), band_diff(value=3.0).astype(np.uint8))
    cv2.imwrite(str(floats / "a.png"), band_diff(value=1.0).astype(np.uint8))
    np.save(diffs / "b.npy", band_diff(value=3.0))
    np.save(floats / "b.npy", band_diff(value=1.0))
    code, rows = run_compare(tmp_path, images, diffs, floats, "--dilate", "2")
    out = capsys.readouterr()
    assert code == 0
    assert rows["a"]["diff_kind"] == "abs" and rows["b"]["diff_kind"] == "signed"
    assert rows["TOTAL"]["diff_kind"] == "mixed"
    assert f(rows["a"], "delta_sum_abs_diff_on_edges") == 2 * 4 * 48
    assert ("warning: 1 of 2 pair(s) hold a plain |diff| file" in out.err
            and "only the delta_* columns are meaningful" in out.err)
    assert "\nTOTAL |dq - df| (mixed) | " in out.out


def test_all_signed_runs_print_no_abs_warning(tmp_path, capsys):
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff(value=3.0))
    np.save(floats / "a.npy", band_diff(value=1.0))
    code, _ = run_compare(tmp_path, images, diffs, floats, "--dilate", "2")
    out = capsys.readouterr()
    assert code == 0 and "plain |diff|" not in out.err
    lines = out.out.splitlines()
    assert lines[-4].startswith("TOTAL | edge_pixels 240")
    assert lines[-3] == ("TOTAL float | sum |dL| on edges 192 | mean on edges 0.8 "
                         "| sum |dL| all 192 | edge/all 1.0000")
    assert lines[-2] == ("TOTAL delta quant - float | sum |dL| on edges 384 | "
                         "mean on edges 1.6 | sum |dL| all 384 | edge/all 0.0000")
    assert lines[-1] == ("TOTAL |dq - df| (signed) | sum |dL| on edges 384 | "
                         "mean on edges 1.6 | sum |dL| all 384 | edge/all 1.0000")


def test_a_pair_in_one_run_only_is_an_error_unless_allowed(tmp_path, capsys):
    images, diffs, floats = compare_dirs(tmp_path)
    for stem in ("a", "b", "c"):
        cv2.imwrite(str(images / f"{stem}.png"), step_image())
        np.save(diffs / f"{stem}.npy", band_diff())
    np.save(floats / "a.npy", band_diff(value=1.0))
    np.save(floats / "b.npy", band_diff(value=1.0))
    # c has no float diff: same-stem pairing of --diffs-float lists it as an
    # image without a diff there; it is in --diffs only.
    code, rows = run_compare(tmp_path, images, diffs, floats)
    err = capsys.readouterr().err
    assert code == 2 and rows is None
    assert "image without a diff in --diffs-float: c" in err
    code, rows = run_compare(tmp_path, images, diffs, floats, "--allow-unpaired")
    out = capsys.readouterr()
    assert code == 0 and set(rows) == {"a", "b", "TOTAL"}
    assert "pair only in --diffs: c" in out.err
    assert "note: 1 pair(s) in one run only, skipped in both: c" in out.out


def test_a_float_pair_missing_from_quant_is_named(tmp_path, capsys):
    images, diffs = diff_name_folder(tmp_path)
    floats = tmp_path / "floats"
    floats.mkdir()
    np.save(floats / f"{AB}.npy", band_diff(value=1.0))
    np.save(floats / f"{B}_{A}_diff_L.npy", band_diff(value=1.0))
    code, rows = run_compare(tmp_path, images, diffs, floats)
    err = capsys.readouterr().err
    assert code == 2 and rows is None
    assert f"pair only in --diffs-float: {B}_{A}_diff_L" in err
    assert "0 pair(s) only in --diffs and 1 only in --diffs-float" in err


def test_compare_pairs_real_diff_names_and_says_the_float_pairing(tmp_path, capsys):
    images, diffs = diff_name_folder(tmp_path)
    floats = tmp_path / "floats"
    floats.mkdir()
    np.save(floats / f"{AB}.npy", band_diff(value=1.0))
    code, rows = run_compare(tmp_path, images, diffs, floats, "--dilate", "2")
    out = capsys.readouterr().out
    assert code == 0 and set(rows) == {AB, "TOTAL"}
    assert f(rows[AB], "qerr_sum_abs_diff_on_edges") == 2 * 4 * 48
    assert "pairing: from <A>_<B>_diff_L names" in out
    assert "pairing in --diffs-float: from <A>_<B>_diff_L names\n" in out   # its edges: none


def test_runs_of_another_shape_are_an_error_naming_the_pair(tmp_path, capsys):
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff())
    np.save(floats / "a.npy", np.zeros((40, W), np.float32))
    code, rows = run_compare(tmp_path, images, diffs, floats)
    err = capsys.readouterr().err
    assert code == 2 and rows is None
    assert "a: --diffs a.npy is 48x64 but --diffs-float a.npy is 40x64" in err


@pytest.mark.parametrize("make_float, message", [
    (lambda p: cv2.imwrite(str(p / "a.png"), np.zeros((48, W), np.uint16)),
     "a: --diffs a.png is uint8 but --diffs-float a.png is uint16"),
    (lambda p: np.save(p / "a.npy", np.zeros((48, W), np.float64)), None),   # npy vs png: fine
])
def test_runs_stored_in_two_bit_depths_are_an_error(tmp_path, capsys, make_float, message):
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    cv2.imwrite(str(diffs / "a.png"), np.zeros((48, W), np.uint8))
    make_float(floats)
    code, _ = run_compare(tmp_path, images, diffs, floats)
    if message is None:
        assert code == 0
    else:
        assert code == 2 and message in capsys.readouterr().err


def test_npy_runs_of_two_dtypes_are_an_error(tmp_path, capsys):
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff().astype(np.float32))
    np.save(floats / "a.npy", band_diff().astype(np.float64))
    code, _ = run_compare(tmp_path, images, diffs, floats)
    assert code == 2
    assert "is float32 but --diffs-float a.npy is float64" in capsys.readouterr().err


def test_diff_offset_float_without_diffs_float_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        K.main(["--images", str(tmp_path), "--diffs", str(tmp_path),
                "--out", str(tmp_path / "o.csv"), "--diff-offset-float", "128"])


def run_diff(values, signed=True):
    values = np.array(values, np.float64)
    return K.RunDiff("x.npy", values, signed, values.dtype, False)


def test_compare_row_and_total_by_hand():
    # Band = the left column. quant [[2, 1], [-4, 0]], float [[1, 1], [1, 3]].
    mask = np.array([[True, False], [True, False]])
    row = K.compare_row("p", mask, run_diff([[2, 1], [-4, 0]]), run_diff([[1, 1], [1, 3]]))
    assert (row["sum_abs_diff_on_edges"], row["sum_abs_diff_all"]) == (6, 7)
    assert (row["float_sum_abs_diff_on_edges"], row["float_sum_abs_diff_all"]) == (2, 6)
    assert row["delta_sum_abs_diff_on_edges"] == 6 - 2
    assert row["delta_mean_abs_diff_on_edges"] == 6 / 2 - 2 / 2
    # dq - df = [[1, 0], [-5, -3]]: on the band |1| + |-5| = 6, all 9
    # (|dq| - |df| would give 1 + 3 = 4 there: the sign counts).
    assert (row["qerr_sum_abs_diff_on_edges"], row["qerr_sum_abs_diff_all"]) == (6, 9)
    assert row["qerr_ratio_edge_to_all"] == 6 / 9
    assert row["diff_kind"] == "signed"
    other = K.compare_row("q", np.ones((2, 2), bool), run_diff([[1, 1], [1, 1]], signed=False),
                          run_diff([[0, 0], [0, 0]]))
    assert other["diff_kind"] == "abs"                          # quant plain
    float_plain = K.compare_row("r", mask, run_diff([[1, 1], [1, 1]]),
                                run_diff([[0, 0], [0, 0]], signed=False))
    assert float_plain["diff_kind"] == "abs"                    # float plain
    total = K.compare_total_row([row, other])
    assert total["edge_pixels"] == 2 + 4
    assert total["qerr_sum_abs_diff_on_edges"] == 6 + 4
    assert total["qerr_mean_abs_diff_on_edges"] == 10 / 6       # not (3 + 1) / 2
    assert total["float_mean_abs_diff_on_edges"] == 2 / 6
    assert total["delta_mean_abs_diff_on_edges"] == (6 + 4) / 6 - 2 / 6
    assert total["diff_kind"] == "mixed"


def test_compare_image_of_another_size_than_both_diffs_is_a_named_error(tmp_path, capsys):
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())                 # 48x64
    np.save(diffs / "a.npy", np.zeros((40, W), np.float32))
    np.save(floats / "a.npy", np.zeros((40, W), np.float32))
    code, rows = run_compare(tmp_path, images, diffs, floats)
    assert code == 2 and rows is None
    assert "error: a: image a.png is 48x64 but diff a.npy is 40x64" in capsys.readouterr().err


def test_diff_offset_float_none_reads_a_plain_float_run_against_a_signed_quant(tmp_path,
                                                                             capsys):
    # The reviewer's case: quant png signed with offset 128 (+3 on the band),
    # float png a plain |diff| (1 on the band). With none the float run gets no
    # offset: its KPI is the stored 1, diff_kind abs, and the run warns.
    images, diffs, floats = compare_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    offset_png(diffs / "a.png", 128, 3)
    cv2.imwrite(str(floats / "a.png"), band_diff(value=1.0).astype(np.uint8))
    code, rows = run_compare(tmp_path, images, diffs, floats, "--dilate", "2",
                             "--diff-offset", "128", "--diff-offset-float", "none")
    out = capsys.readouterr()
    assert code == 0
    r = rows["a"]
    assert f(r, "sum_abs_diff_all") == 3 * 4 * 48
    assert f(r, "float_sum_abs_diff_all") == 1 * 4 * 48
    assert r["diff_kind"] == "abs"
    assert "warning: 1 of 1 pair(s) hold a plain |diff| file" in out.err
    assert "| --diff-offset-float none\n" in out.out


# 1e400 overflows to inf in float(). "=" keeps argparse from reading -inf as a flag.
@pytest.mark.parametrize("flag", ["--diff-offset", "--diff-offset-float"])
@pytest.mark.parametrize("value", ["abc", "nan", "NaN", "inf", "-inf", "Infinity", "1e400"])
def test_a_bad_offset_is_refused_saying_what_it_accepts(tmp_path, capsys, flag, value):
    with pytest.raises(SystemExit) as refused:
        K.main(["--images", str(tmp_path), "--diffs", str(tmp_path), "--diffs-float",
                str(tmp_path), "--out", str(tmp_path / "o.csv"), f"{flag}={value}"])
    assert refused.value.code == 2
    assert (f"argument {flag}: expected a finite number or 'none', got {value!r}"
            in capsys.readouterr().err)


def test_diff_offset_none_is_no_offset(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    cv2.imwrite(str(diffs / "a.png"), np.full((48, W), 7, np.uint8))
    code, rows = run_cli(tmp_path, images, diffs, "--diff-offset", "None")
    assert code == 0
    assert float(rows["a"]["sum_abs_diff_all"]) == 7 * 48 * W
    assert "| --diff-offset none\n" in capsys.readouterr().out


# ---------------------------------------------- --mv: motion vectors
# DOF infer's files, as the owner's probe measured one (2026-11-12): <A>_<B>.dof,
# int16 little-endian, planar (u plane then v plane), no header, 1/4 of the frame.
EPE_SCALE = {"int16": 4, "int32": 4, "float16": 1, "float32": 1, "float64": 1}


def write_dof(path, flow, dtype="int16", layout="2hw", header=b""):
    """flow (H, W, 2) in stored units, written as DOF infer would."""
    data = flow.transpose(2, 0, 1) if layout == "2hw" else flow
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(np.ascontiguousarray(data).astype(np.dtype(dtype).newbyteorder("<")).tobytes())


def mv_dirs(tmp_path):
    """The A/B step folder of diff_name_folder, and empty mv/ and mv_float/."""
    images, diffs = diff_name_folder(tmp_path)
    mv, mvf = tmp_path / "mv", tmp_path / "mv_float"
    mv.mkdir()
    mvf.mkdir()
    return images, diffs, mv, mvf


def run_mv(tmp_path, images, diffs, mv, mvf, *extra):
    return run_cli(tmp_path, images, diffs, "--mv", str(mv), "--mv-float", str(mvf), *extra)


def by_hand_flows(h=48, scale=1):
    """float flow (1.5, -2) everywhere: |flow| 2.5. quant adds (3, 4) on
    columns 30..33 (EPE 5, inside any 2-px band); off the band (0, 0.75) on
    0..9, (1, 0) on 10..11, (0, 0.5) on 12..13 and (0.25, 0) on 50..63: EPE
    0.75, exactly 1 and 0.5 (not above their thresholds) and 0.25. All exact
    in every dtype: integer types store them x 4."""
    flt = np.empty((h, W, 2))
    flt[..., 0], flt[..., 1] = 1.5, -2.0
    quant = flt.copy()
    quant[:, BAND] += (3.0, 4.0)
    quant[:, :10, 1] += 0.75
    quant[:, 10:12, 0] += 1.0
    quant[:, 12:14, 1] += 0.5
    quant[:, 50:, 0] += 0.25
    return quant * scale, flt * scale


@pytest.mark.parametrize("layout", ["hw2", "2hw"])
@pytest.mark.parametrize("dtype", sorted(EPE_SCALE))
def test_epe_by_hand_in_every_dtype_and_layout(tmp_path, capsys, dtype, layout):
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    quant, flt = by_hand_flows(scale=EPE_SCALE[dtype])
    write_dof(mv / f"{A}_{B}.dof", quant, dtype, layout)
    write_dof(mvf / f"{A}_{B}.dof", flt, dtype, layout)
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf, "--dilate", "2",
                        "--mv-dtype", dtype, "--mv-layout", layout, "--mv-shape", "48", "64",
                        "--mv-unit-res", "full", "--mv-scale", str(EPE_SCALE[dtype]))
    assert code == 0
    r = rows[AB]
    assert (int(r["mv_H"]), int(r["mv_W"]), int(r["mv_pixels"])) == (48, W, 48 * W)
    assert int(r["mv_edge_pixels"]) == 240                      # the 5-column band
    assert f(r, "mv_epe_sum_on_edges") == 5 * 4 * 48            # 960
    assert f(r, "mv_epe_mean_on_edges") == 960 / 240
    assert int(r["mv_n_epe_over_0.5px_on_edges"]) == 4 * 48
    assert f(r, "mv_frac_epe_over_0.5px_on_edges") == 192 / 240
    assert int(r["mv_n_epe_over_1px_on_edges"]) == 4 * 48
    assert f(r, "mv_float_mag_sum_on_edges") == 2.5 * 240
    assert f(r, "mv_float_mag_mean_on_edges") == 2.5
    assert f(r, "mv_rel_epe_on_edges") == 960 / 600
    all_sum = 960 + (0.75 * 10 + 1 * 2 + 0.5 * 2 + 0.25 * 14) * 48   # 1632
    assert f(r, "mv_epe_sum_all") == all_sum
    assert f(r, "mv_epe_mean_all") == pytest.approx(all_sum / (48 * W))
    assert int(r["mv_n_epe_over_0.5px_all"]) == 192 + (10 + 2) * 48   # EPE 5, 0.75, 1
    assert int(r["mv_n_epe_over_1px_all"]) == 192                     # EPE 5 only
    assert f(r, "mv_rel_epe_all") == pytest.approx(all_sum / (2.5 * 48 * W))
    assert "no --mv-scale" not in capsys.readouterr().err


REAL_A, REAL_B = "0001_IMX874_Front_001", "0001_IMX874_Front_000"


def test_the_real_file_shape_with_default_flags_reduces_the_band_to_the_quarter_grid(
        tmp_path, capsys):
    # 576 x 1024 frames, step at column 512; the flow 144 x 256 int16 planar,
    # 147456 bytes like the owner's file. Band (--dilate 2) is 5 frame columns
    # inside 509..514, so on the 1/4 grid columns 127 and 128 (509 // 4, 514 // 4):
    # a cell is on the band when any of its 4 x 4 frame pixels is. EPE 50 raw
    # on exactly those cells, 0 elsewhere.
    images, diffs = make_dirs(tmp_path)
    frame = np.full((576, 1024), 60, np.uint8)
    frame[:, 512:] = 180
    cv2.imwrite(str(images / f"{REAL_A}.png"), frame)
    cv2.imwrite(str(images / f"{REAL_B}.png"), np.full((576, 1024), 90, np.uint8))
    np.save(diffs / f"{REAL_A}_{REAL_B}_diff_L.npy", np.zeros((576, 1024), np.float32))
    mv, mvf = tmp_path / "mv", tmp_path / "mv_float"
    mv.mkdir()
    mvf.mkdir()
    flt = np.empty((144, 256, 2))
    flt[..., 0], flt[..., 1] = 8, -4
    quant = flt.copy()
    quant[:, 127:129] += (30, 40)
    write_dof(mv / f"{REAL_A}_{REAL_B}.dof", quant)
    write_dof(mvf / f"{REAL_A}_{REAL_B}.dof", flt)
    assert os.path.getsize(mv / f"{REAL_A}_{REAL_B}.dof") == 147456
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf, "--dilate", "2")
    out = capsys.readouterr()
    assert code == 0
    r = rows[f"{REAL_A}_{REAL_B}_diff_L"]
    assert (int(r["mv_H"]), int(r["mv_W"])) == (144, 256)
    assert int(r["mv_edge_pixels"]) == 2 * 144
    assert f(r, "mv_epe_sum_on_edges") == f(r, "mv_epe_sum_all") == 50 * 2 * 144
    assert f(r, "mv_rel_epe_on_edges") == pytest.approx(50 / math.hypot(8, 4))
    assert ("mv as read: --mv-dtype int16 (little-endian) | --mv-layout 2hw | "
            "--mv-header-bytes 0 | --mv-scale 1 | --mv-shape 1/4 of the reference frame "
            "| --mv-unit-res quarter | files <A>_<B>.dof\n") in out.out
    assert "TOTAL mv EPE on_edges (quarter grid) | pixels 288 | sum 14400 | mean 50 raw units" in out.out
    assert "warning: no --mv-scale: EPE is in the raw stored units" in out.err


def test_band_at_a_lower_size_keeps_a_cell_any_frame_pixel_of_which_is_on_it():
    mask = np.zeros((8, 8), bool)
    mask[5, 6] = True                       # in the 4 x 4 block (1, 1) only
    assert np.array_equal(K.band_at(mask, (2, 2)), [[False, False], [False, True]])
    assert np.array_equal(K.band_at(mask, (8, 8)), mask)


def test_half_size_flow_and_its_total(tmp_path):
    # Flow at 1/2: 24 x 32. EPE 5 on cells 15..16 (frame columns 30..33); the
    # band (5 frame columns in 29..34) covers 3 cells. Two equal pairs: the
    # TOTAL sums the counts.
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    cv2.imwrite(str(images / "0115_IMX874_Front_143.png"), step_image())
    np.save(diffs / f"0115_IMX874_Front_143_{A}_diff_L.npy", band_diff())
    flt = np.zeros((24, 32, 2))
    quant = flt.copy()
    quant[:, 15:17] = (3, 4)
    for stem in (f"{A}_{B}", f"0115_IMX874_Front_143_{A}"):
        write_dof(mv / f"{stem}.dof", quant, "float32")
        write_dof(mvf / f"{stem}.dof", flt, "float32")
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf, "--dilate", "2", "--mv-dtype",
                        "float32", "--mv-shape", "24", "32", "--mv-unit-res", "full")
    assert code == 0
    assert int(rows[AB]["mv_edge_pixels"]) == 3 * 24
    assert f(rows[AB], "mv_epe_sum_on_edges") == 5 * 2 * 24
    t = rows["TOTAL"]
    assert t["mv_H"] == "" and int(t["mv_edge_pixels"]) == 2 * 72
    assert int(t["mv_pixels"]) == 2 * 24 * 32
    assert f(t, "mv_epe_sum_on_edges") == 2 * 240
    assert f(t, "mv_epe_mean_on_edges") == 480 / 144
    assert f(t, "mv_frac_epe_over_1px_on_edges") == 2 * 48 / 144
    assert math.isnan(f(t, "mv_rel_epe_on_edges"))           # float flow all zero


def test_wrong_size_is_refused_with_the_sizes_that_would_match(tmp_path, capsys):
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    flow = np.zeros((12, 16, 2))                                  # 1/4, int16: 768 bytes
    write_dof(mv / f"{A}_{B}.dof", flow)
    write_dof(mvf / f"{A}_{B}.dof", flow)
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf, "--mv-dtype", "float32")
    err = capsys.readouterr().err
    assert code == 2 and rows is None
    assert (f"{A}_{B}.dof: 768 bytes, but --mv-dtype float32 at 12x16 with "
            f"--mv-header-bytes 0 needs 1536") in err
    assert "  int16    1/4    12x16             768  yes\n" in err
    assert "  float16  1/4    12x16             768  yes\n" in err
    assert "  float32  1/2    24x32            6144  -\n" in err
    assert "--mv-layout does not change the size" in err


def test_a_header_is_offered_when_the_size_is_a_little_above_the_data(tmp_path, capsys):
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    flow = np.zeros((12, 16, 2))
    write_dof(mv / f"{A}_{B}.dof", flow, header=b"PIEH" + bytes(8))
    write_dof(mvf / f"{A}_{B}.dof", flow, header=b"PIEH" + bytes(8))
    code, _ = run_mv(tmp_path, images, diffs, mv, mvf)
    err = capsys.readouterr().err
    assert code == 2
    assert "  int16    1/4    12x16             768  with --mv-header-bytes 12\n" in err
    assert "starts with PIEH, as a Middlebury .flo does" in err
    code, _ = run_mv(tmp_path, images, diffs, mv, mvf, "--mv-header-bytes", "12")
    assert code == 0


def test_an_mv_folder_without_dof_files_is_an_error(tmp_path, capsys):
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    write_dof(mvf / f"{A}_{B}.dof", np.zeros((12, 16, 2)))
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf)
    assert code == 2 and rows is None
    assert f"--mv: no .dof file in {mv}" in capsys.readouterr().err


def test_nan_in_a_flow_is_refused(tmp_path, capsys):
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    flow = np.zeros((12, 16, 2))
    write_dof(mvf / f"{A}_{B}.dof", flow, "float32")
    flow[3, 4, 1] = np.nan
    write_dof(mv / f"{A}_{B}.dof", flow, "float32")
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf, "--mv-dtype", "float32")
    assert code == 2 and rows is None
    assert (f"{A}_{B}.dof: 1 motion vector value(s) are NaN or inf"
            in capsys.readouterr().err)


@pytest.mark.parametrize("shape, unit, message", [
    (("10", "16"), "full", "mv shape 10x16 is not the frame 48x64 divided by one integer"),
    (("96", "128"), "full", "mv shape 96x128 is not the frame 48x64"),
    (("24", "32"), "quarter", "--mv-unit-res quarter, but the flow is 1/2 of the frame"),
])
def test_a_flow_size_that_is_no_integer_division_or_unit_is_refused(tmp_path, capsys, shape,
                                                                    unit, message):
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    flow = np.zeros((int(shape[0]), int(shape[1]), 2))
    write_dof(mv / f"{A}_{B}.dof", flow)
    write_dof(mvf / f"{A}_{B}.dof", flow)
    code, _ = run_mv(tmp_path, images, diffs, mv, mvf, "--mv-shape", *shape,
                     "--mv-unit-res", unit)
    assert code == 2 and message in capsys.readouterr().err


def test_unmatched_mv_files_and_pairs_are_listed_and_an_error_unless_allowed(tmp_path,
                                                                            capsys):
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    cv2.imwrite(str(images / "0115_IMX874_Front_143.png"), step_image())
    np.save(diffs / f"0115_IMX874_Front_143_{A}_diff_L.npy", band_diff())
    flow = np.zeros((12, 16, 2))
    for folder in (mv, mvf):
        write_dof(folder / f"{A}_{B}.dof", flow)
    write_dof(mv / f"0115_IMX874_Front_143_{A}.dof", flow)    # none in mv_float
    write_dof(mv / "zzz.dof", flow)                           # names no pair
    write_dof(mvf / f"{B}_{A}.dof", flow)                     # a pair with no diff
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf)
    err = capsys.readouterr().err
    assert code == 2 and rows is None
    assert "mv file in --mv naming no pair: zzz" in err
    assert f"mv file in --mv-float of a pair with no diff: {B}_{A}.dof" in err
    assert f"pair without an mv file in --mv-float: 0115_IMX874_Front_143_{A}_diff_L" in err
    assert "2 mv file(s) match no diff pair and 1 pair(s) miss an mv file" in err
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf, "--allow-unpaired")
    out = capsys.readouterr().out
    assert code == 0 and set(rows) == {AB, "TOTAL"}            # skipped in every measure
    assert (f"note: 1 pair(s) without both mv files, skipped: "
            f"0115_IMX874_Front_143_{A}_diff_L") in out


def test_a_name_suffix_is_read_and_printed_and_two_namings_are_refused(tmp_path, capsys):
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    flow = np.zeros((12, 16, 2))
    for folder in (mv, mvf):
        write_dof(folder / f"{A}_{B}_mv.dof", flow)
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf)
    assert code == 0 and int(rows[AB]["mv_H"]) == 12
    assert "| files <A>_<B>_mv.dof\n" in capsys.readouterr().out
    write_dof(mv / f"{A}_{B}.dof", flow)
    code, _ = run_mv(tmp_path, images, diffs, mv, mvf)
    assert code == 2 and "--mv: two mv files for the pair" in capsys.readouterr().err
    os.remove(mv / f"{A}_{B}.dof")
    cv2.imwrite(str(images / "0115_IMX874_Front_143.png"), step_image())
    np.save(diffs / f"0115_IMX874_Front_143_{A}_diff_L.npy", band_diff())
    write_dof(mv / f"0115_IMX874_Front_143_{A}_flow.dof", flow)
    code, _ = run_mv(tmp_path, images, diffs, mv, mvf, "--allow-unpaired")
    assert code == 2
    assert "--mv: the mv file names end in 2 suffixes ('flow', 'mv')" in capsys.readouterr().err


def test_an_ambiguous_mv_name_is_refused(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    for stem in ("a", "b", "b_x"):
        cv2.imwrite(str(images / f"{stem}.png"), step_image())
    np.save(diffs / "a_b_diff_L.npy", band_diff())
    mv, mvf = tmp_path / "mv", tmp_path / "mv_float"
    mv.mkdir()
    mvf.mkdir()
    for folder in (mv, mvf):
        write_dof(folder / "a_b_x.dof", np.zeros((12, 16, 2)))   # a + b_x, or a + b + "x"
    code, _ = run_mv(tmp_path, images, diffs, mv, mvf)
    assert code == 2
    assert "mv file 'a_b_x': ambiguous, it reads as the pair of a + b or a + b_x" in (
        capsys.readouterr().err)


def test_same_stem_pairing_pairs_mv_files_by_the_stem(tmp_path):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff())
    mv, mvf = tmp_path / "mv", tmp_path / "mv_float"
    mv.mkdir()
    mvf.mkdir()
    write_dof(mv / "a.dof", np.ones((12, 16, 2)))
    write_dof(mvf / "a.dof", np.zeros((12, 16, 2)))
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf, "--mv-scale", "1")
    assert code == 0
    assert f(rows["a"], "mv_epe_sum_all") == pytest.approx(math.sqrt(2) * 12 * 16)


def test_mv_with_diffs_float_adds_the_mv_columns_after_the_compare_ones(tmp_path):
    images, diffs, mv, mvf = mv_dirs(tmp_path)
    floats = tmp_path / "floats"
    floats.mkdir()
    np.save(floats / f"{AB}.npy", band_diff(value=1.0))
    for folder in (mv, mvf):
        write_dof(folder / f"{A}_{B}.dof", np.zeros((12, 16, 2)))
    code, rows = run_mv(tmp_path, images, diffs, mv, mvf, "--diffs-float", str(floats))
    assert code == 0
    with open(tmp_path / "kpi.csv", newline="") as fh:
        assert next(csv.reader(fh)) == COMPARE_HEADER + list(K.MV_FIELDS)
    assert f(rows[AB], "qerr_sum_abs_diff_all") == 2 * 4 * 48


@pytest.mark.parametrize("argv, message", [
    (["--mv", "x"], "--mv needs --mv-float"),
    (["--mv-float", "x"], "--mv-float needs --mv"),
    (["--mv-dtype", "int16"], "--mv-dtype needs --mv"),
    (["--mv-probe"], "--mv-probe needs --mv"),
    (["--mv", "x", "--mv-float", "y", "--mv-scale", "0"],
     "argument --mv-scale: expected a finite number other than 0, got '0'"),
    (["--mv", "x", "--mv-float", "y", "--mv-scale=nan"],
     "argument --mv-scale: expected a finite number other than 0, got 'nan'"),
    (["--mv", "x", "--mv-float", "y", "--mv-header-bytes=-1"],
     "argument --mv-header-bytes: expected an integer >= 0, got '-1'"),
    (["--mv", "x", "--mv-float", "y", "--mv-shape", "0", "4"],
     "argument --mv-shape: expected an integer >= 1, got '0'"),
])
def test_bad_mv_flags_are_refused(tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit) as refused:
        K.main(["--images", str(tmp_path), "--diffs", str(tmp_path),
                "--out", str(tmp_path / "o.csv"), *argv])
    assert refused.value.code == 2
    assert message in capsys.readouterr().err


def test_diffs_and_out_stay_required_without_the_probe(tmp_path, capsys):
    with pytest.raises(SystemExit) as refused:
        K.main(["--images", str(tmp_path)])
    assert refused.value.code == 2
    assert "the following arguments are required: --diffs, --out" in capsys.readouterr().err


def test_probe_prints_what_fits_and_writes_nothing(tmp_path, capsys):
    # An interleaved (hw2) int16 flow at 1/4, u -60 and v 500 everywhere: read
    # as hw2 each plane is flat (step 0); read as 2hw, plane 0 alternates
    # u, v, u, ... (step |500 - -60| = 560, median (500 - 60) / 2 = 220).
    images, _, mv, _ = mv_dirs(tmp_path)
    flow = np.zeros((12, 16, 2))
    flow[..., 0], flow[..., 1] = -60, 500
    write_dof(mv / f"{A}_{B}.dof", flow, layout="hw2")
    before = sorted(p.name for p in tmp_path.rglob("*"))
    code = K.main(["--images", str(images), "--mv", str(mv), "--mv-probe"])
    out = capsys.readouterr().out
    assert code == 0
    assert sorted(p.name for p in tmp_path.rglob("*")) == before
    assert f"{A}_{B}.dof, 768 bytes" in out
    assert "  int16    1/4    12x16             768  yes\n" in out
    assert "int16 12x16 header 0: 0 non-finite value(s), as stored (before --mv-scale)" in out
    assert "  hw2: u min -60 max -60 median -60 step 0 | v min 500 max 500 median 500 step 0" in out
    assert "  2hw: u min -60 max 500 median 220 step 560 | v min -60 max 500 median 220 step 560" in out
    assert "probe only: nothing measured, nothing written" in out
