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
    raw = K.load_diff_raw(str(diffs / f"{AB}.jpg"))
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
        assert K.load_diff_raw(str(path)).shape == shape[:2]


def test_3_channel_diff_within_tolerance_takes_channel_0(tmp_path):
    path = tmp_path / "d.png"
    diff = np.full((8, 8, 3), 10, np.uint8)
    diff[..., 1] = 12                                       # +2: JPEG-like noise
    diff[..., 2] = 9
    cv2.imwrite(str(path), diff)
    raw = K.load_diff_raw(str(path), channel_tol=3)
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


# ---------------------------------------------- no --diffs-float: unchanged
# The output of e026830 (the tool before --diffs-float), run on these two
# folders and pasted here: CSV bytes, stdout (the CSV path as <OUT>), stderr.
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
    cv2.imwrite(str(images / "a.png"), step_image())
    cv2.imwrite(str(images / "b.png"), step_image(24))
    stored = np.full((48, W), 128, np.uint8)
    stored[:, 30:32], stored[:, 32:34], stored[:, :4] = 123, 131, 121
    cv2.imwrite(str(diffs / "a.png"), stored)
    np.save(diffs / "b.npy", band_diff(24, -1.5))
    return images, diffs, ["--detector", "sobel", "--diff-offset", "128"]


@pytest.mark.parametrize("scenario", sorted(PIN))
def test_without_diffs_float_the_output_is_byte_identical_to_e026830(tmp_path, capsys,
                                                                   scenario):
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
    assert "pairing in --diffs-float: from <A>_<B>_diff_L names" in out


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
    assert other["diff_kind"] == "abs"
    total = K.compare_total_row([row, other])
    assert total["edge_pixels"] == 2 + 4
    assert total["qerr_sum_abs_diff_on_edges"] == 6 + 4
    assert total["qerr_mean_abs_diff_on_edges"] == 10 / 6       # not (3 + 1) / 2
    assert total["float_mean_abs_diff_on_edges"] == 2 / 6
    assert total["delta_mean_abs_diff_on_edges"] == (6 + 4) / 6 - 2 / 6
    assert total["diff_kind"] == "mixed"
