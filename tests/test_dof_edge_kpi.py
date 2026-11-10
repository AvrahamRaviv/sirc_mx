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

cv2 = pytest.importorskip("cv2", reason="optional eval extra (opencv) not installed")

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
    pairs = K.pair_files(str(images), str(diffs))
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


# ---------------------------------------------- the DOF tool's files (t50)
# Names of the real shape, synthetic content: <A>_<B>_diff_L.jpg, A and B
# image stems that hold underscores themselves.
A, B = "0115_IMX874_Front_136", "0115_IMX874_Front_135"
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


def test_diff_name_default_reference_is_second_and_said(tmp_path, capsys):
    images, diffs = diff_name_folder(tmp_path)
    code, rows = run_cli(tmp_path, images, diffs, "--dilate", "2")
    assert code == 0
    assert int(rows[AB]["edge_pixels"]) == 0               # B, the flat frame
    assert "edges on the second frame (B, --ref-frame second)" in capsys.readouterr().out


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


def test_forced_diff_name_pairing_refuses_another_name(tmp_path, capsys):
    images, diffs = make_dirs(tmp_path)
    cv2.imwrite(str(images / "a.png"), step_image())
    np.save(diffs / "a.npy", band_diff())
    code, _ = run_cli(tmp_path, images, diffs, "--pair-from-diff-name")
    assert code == 2 and "does not end in _diff_L" in capsys.readouterr().err
