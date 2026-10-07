"""Topic A - sweep calibration drift (yaw, translation) và đo mismatch của projection LiDAR -> ảnh.

Với mỗi object GT (Car/Van/Truck/Pedestrian/Cyclist):
  - Điểm thuộc object = điểm LiDAR nằm trong 3D box GT, xác định bằng calib GỐC (không đổi giữa các mức).
  - Chiếu đúng tập điểm đó bằng calib đã lệch rồi đếm % điểm rơi vào 2D box của label.
  - Chỉ thay đúng một yếu tố (calib); frame, object, seed giữ nguyên. Phép perturb là tất định.

Chạy:
    python src/calib_sweep.py
    python src/calib_sweep.py --data-root data/nuscenes_mini_subset --frames scene-0103_010 --out results/sweep_nusc.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.datasets import load_frame  # noqa: E402
from starter.projection import (cam_to_image, perturb_extrinsic,  # noqa: E402
                                project_velo_to_image, velo_to_cam)

DEFAULT_FRAMES = ["000004", "000007", "000009", "000010", "000012",
                  "000019", "000025", "000031", "000032", "000049"]
CLASSES = {"Car", "Van", "Truck", "Pedestrian", "Cyclist"}
DIST_BINS = [(0, 15), (15, 30), (30, 50), (50, 200)]


def points_in_box3d(points_xyz: np.ndarray, calib, obj) -> np.ndarray:
    """Mask (N,) điểm velodyne nằm trong 3D box của obj (box ở rectified camera frame)."""
    pc = velo_to_cam(points_xyz, calib) - obj.location
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    # xoay ngược rotation_y quanh trục y của camera (R^T @ p)
    x = c * pc[:, 0] - s * pc[:, 2]
    z = s * pc[:, 0] + c * pc[:, 2]
    h, w, l = obj.dimensions
    return (np.abs(x) <= l / 2) & (np.abs(z) <= w / 2) & (pc[:, 1] <= 0) & (pc[:, 1] >= -h)


def raw_uv(points_xyz: np.ndarray, calib) -> tuple[np.ndarray, np.ndarray]:
    """uv không lọc theo ảnh (chỉ lọc depth > 0.1) để đo độ dịch pixel."""
    cam = velo_to_cam(points_xyz, calib)
    ok = cam[:, 2] > 0.1
    h = np.hstack([cam, np.ones((len(cam), 1))]) @ calib.P2.T
    uv = h[:, :2] / np.where(ok, h[:, 2], 1.0)[:, None]
    return uv, ok


def build_configs(yaws: list[float], trans_cm: list[float]) -> list[dict]:
    cfgs = [dict(ptype="baseline", axis="-", value=0.0)]
    cfgs += [dict(ptype="yaw_deg", axis="z", value=y) for y in yaws if y != 0]
    for ax in ("x", "y", "z"):
        cfgs += [dict(ptype="trans_cm", axis=ax, value=t) for t in trans_cm]
    return cfgs


def perturbed(calib, cfg):
    if cfg["ptype"] == "baseline":
        return calib
    if cfg["ptype"] == "yaw_deg":
        return perturb_extrinsic(calib, yaw_deg=cfg["value"])
    t = [0.0, 0.0, 0.0]
    t["xyz".index(cfg["axis"])] = cfg["value"] / 100.0
    return perturb_extrinsic(calib, t_xyz_m=tuple(t))


def dist_bin(d: float) -> str:
    for lo, hi in DIST_BINS:
        if lo <= d < hi:
            return f"{lo}-{hi}m" if hi < 200 else f">{lo}m"
    return "other"


def run(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(args.seed)  # sweep là tất định; seed ghi lại cho đủ quy ước
    cfgs = build_configs(args.yaw_deg, args.trans_cm)
    obj_rows, fov_rows = [], []
    for fid in args.frames:
        fr = load_frame(args.data_root, fid)
        pts = fr["points"][:, :3].astype(np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        calib0, shape = fr["calib"], fr["image"].shape
        objs = []
        for i, o in enumerate(fr["labels"]):
            if o.type not in CLASSES or o.truncated >= 0.5:
                continue
            m = points_in_box3d(pts, calib0, o)
            if m.sum() >= args.min_points:
                objs.append((i, o, pts[m]))
        uv0 = {i: raw_uv(p, calib0) for i, _, p in objs}
        for cfg in cfgs:
            calib = perturbed(calib0, cfg)
            _, _, fmask = project_velo_to_image(pts, calib, shape)
            fov_rows.append(dict(frame=fid, **cfg, pct_in_fov=100 * fmask.mean()))
            for i, o, p in objs:
                uv, _, mask = project_velo_to_image(p, calib, shape)
                x1, y1, x2, y2 = o.bbox
                inbox = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
                uvp, okp = raw_uv(p, calib)
                both = okp & uv0[i][1]
                shift = np.linalg.norm(uvp[both] - uv0[i][0][both], axis=1)
                obj_rows.append(dict(
                    frame=fid, obj_idx=i, type=o.type, dist_m=o.location[2], dist_bin=dist_bin(o.location[2]),
                    bbox_w_px=x2 - x1, bbox_h_px=y2 - y1, n_points=len(p), **cfg,
                    pct_in_box=100 * inbox.sum() / len(p),
                    median_px_shift=float(np.median(shift)) if both.any() else np.nan))
    return pd.DataFrame(obj_rows), pd.DataFrame(fov_rows)


def aggregate(obj: pd.DataFrame, fov: pd.DataFrame) -> pd.DataFrame:
    key = ["ptype", "axis", "value"]
    base = obj[obj.ptype == "baseline"].groupby("dist_bin").pct_in_box.mean()
    all_ = obj.assign(dist_bin="all")
    both = pd.concat([obj, all_])
    base_all = both[both.ptype == "baseline"].groupby("dist_bin").pct_in_box.mean()
    agg = both.groupby(key + ["dist_bin"]).agg(
        n_objects=("pct_in_box", "size"), n_points=("n_points", "sum"),
        pct_in_box_mean=("pct_in_box", "mean"), median_px_shift=("median_px_shift", "median")).reset_index()
    agg["pct_in_box_drop_vs_baseline"] = agg.dist_bin.map(base_all) - agg.pct_in_box_mean
    fov_m = fov.groupby(key).pct_in_fov.mean().reset_index()
    out = agg.merge(fov_m, on=key)
    return out.round(3)


def plot(agg: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [b for b in agg.dist_bin.unique() if b != "all"]
    bins.sort(key=lambda b: float(b.strip(">m").split("-")[0]))
    base = agg[agg.ptype == "baseline"].set_index("dist_bin").pct_in_box_mean
    yaw = agg[agg.ptype == "yaw_deg"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for b in bins:
        d = yaw[yaw.dist_bin == b].sort_values("value")
        if b in base.index:
            ax.plot([0, *d.value], [base[b], *d.pct_in_box_mean], marker="o", label=f"{b}")
    ax.set(xlabel="yaw drift (deg)", ylabel="% điểm object trong 2D box", title="Yaw drift vs % in box theo khoảng cách")
    ax.legend(title="khoảng cách"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(out_dir / "sweep_yaw_pct_in_box.png", dpi=130), plt.close(fig)

    tr = agg[agg.ptype == "trans_cm"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for a, ax_name in zip(axes, "xyz"):
        for b in bins:
            d = tr[(tr.dist_bin == b) & (tr.axis == ax_name)].sort_values("value")
            if b in base.index:
                a.plot([0, *d.value], [base[b], *d.pct_in_box_mean], marker="o", label=b)
        a.set(xlabel=f"t{ax_name} (cm)", title=f"Dịch theo trục {ax_name} của LiDAR"), a.grid(alpha=0.3)
    axes[0].set_ylabel("% điểm object trong 2D box"), axes[0].legend(title="khoảng cách")
    fig.tight_layout(), fig.savefig(out_dir / "sweep_trans_pct_in_box.png", dpi=130), plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep calibration drift (yaw/translation) và đo % điểm LiDAR rơi vào 2D box")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    ap.add_argument("--yaw-deg", nargs="+", type=float, default=[0.5, 1, 2, 3])
    ap.add_argument("--trans-cm", nargs="+", type=float, default=[2, 5, 10])
    ap.add_argument("--min-points", type=int, default=10, help="bỏ object có ít điểm LiDAR hơn ngưỡng này")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/yaw_perturb_sweep.csv")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    obj, fov = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg = aggregate(obj, fov)
    agg.to_csv(out, index=False)
    per_obj = out.with_name(out.stem.replace("_sweep", "") + "_per_object.csv")
    obj.round(3).to_csv(per_obj, index=False)
    print(f"objects={obj.groupby(['frame', 'obj_idx']).ngroups} configs={obj[['ptype', 'axis', 'value']].drop_duplicates().shape[0]}")
    print(f"-> {out}\n-> {per_obj}")
    if not args.no_plot:
        fig_dir = out.parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        plot(agg, fig_dir)
    show = agg[agg.dist_bin.isin(["all"])][["ptype", "axis", "value", "pct_in_box_mean", "pct_in_fov", "median_px_shift"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
