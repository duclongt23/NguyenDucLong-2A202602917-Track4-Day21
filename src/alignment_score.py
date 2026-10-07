"""Topic A (Advanced) - alignment score: khớp depth edge của LiDAR với Canny edge của ảnh.

Ý tưởng: nơi LiDAR nhảy depth đột ngột (mép xe, cột, người) phải trùng với biên trong ảnh.
  1. LiDAR edge: điểm liên tiếp trong scan (cùng beam) có |Δrange| > --depth-jump -> lấy điểm gần hơn.
  2. Ảnh: Canny -> distance transform DT (khoảng cách pixel tới edge ảnh gần nhất).
  3. score = mean(exp(-DT(u, v) / sigma)) trên các LiDAR edge point sau khi chiếu. Càng cao càng khớp.

Sweep yaw 0..3 độ, tính (a) ratio = score/score_baseline của từng frame, (b) ngưỡng tuyệt đối không cần biết baseline
của frame: mean - 2*std của score ở yaw 0 trên tất cả frame. Báo tỉ lệ frame bị phát hiện ở mỗi mức yaw.

    python src/alignment_score.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.datasets import load_frame  # noqa: E402
from starter.projection import perturb_extrinsic, project_velo_to_image  # noqa: E402

DEFAULT_FRAMES = ["000004", "000007", "000009", "000010", "000012",
                  "000019", "000025", "000031", "000032", "000049"]


def lidar_edge_points(points: np.ndarray, depth_jump: float, max_range: float) -> np.ndarray:
    """Điểm velodyne (M, 3) nằm ở mép depth (foreground side). Dựa trên thứ tự scan của file KITTI."""
    xyz = points[:, :3].astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    r = np.linalg.norm(xyz, axis=1)
    dr = np.diff(r)
    # cùng beam: bước azimuth nhỏ (loại điểm nhảy sang vòng quét khác)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    d_az = np.abs(np.diff(az))
    same = (d_az < np.deg2rad(1.0)) & (np.abs(xyz[1:, 2] - xyz[:-1, 2]) < 0.5)
    jump = same & (np.abs(dr) > depth_jump)
    idx = np.where(jump)[0]
    pick = np.where(dr[idx] > 0, idx, idx + 1)  # điểm gần hơn
    pick = pick[r[pick] < max_range]
    return xyz[np.unique(pick)]


def image_dt(image: np.ndarray, lo: int, hi: int) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, lo, hi)
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)


def score(edge_pts: np.ndarray, calib, dt: np.ndarray, sigma: float, shape) -> float:
    uv, _, _ = project_velo_to_image(edge_pts, calib, shape)
    if len(uv) < 20:
        return float("nan")
    u = np.clip(uv[:, 0].astype(int), 0, dt.shape[1] - 1)
    v = np.clip(uv[:, 1].astype(int), 0, dt.shape[0] - 1)
    return float(np.exp(-dt[v, u] / sigma).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Alignment score (LiDAR depth edge vs Canny) và ngưỡng phát hiện calibration drift")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    ap.add_argument("--max-yaw", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=3.0, help="px, độ rộng kernel exp(-DT/sigma)")
    ap.add_argument("--depth-jump", type=float, default=0.8, help="mét, ngưỡng nhảy range để coi là depth edge")
    ap.add_argument("--max-range", type=float, default=40.0, help="chỉ dùng edge point gần hơn ngưỡng này (m)")
    ap.add_argument("--canny", nargs=2, type=int, default=[50, 150])
    ap.add_argument("--ratio-thr", type=float, default=0.9, help="ratio < ngưỡng này là phát hiện drift")
    ap.add_argument("--out", default="results/alignment_score_sweep.csv")
    args = ap.parse_args()

    yaws = np.round(np.arange(0, args.max_yaw + 1e-9, args.step), 3)
    rows = []
    for fid in args.frames:
        fr = load_frame(args.data_root, fid)
        edge = lidar_edge_points(fr["points"], args.depth_jump, args.max_range)
        dt = image_dt(fr["image"], *args.canny)
        for y in yaws:
            c = perturb_extrinsic(fr["calib"], yaw_deg=float(y))
            rows.append(dict(frame=fid, yaw_deg=y, n_edge_pts=len(edge), score=score(edge, c, dt, args.sigma, fr["image"].shape)))
    df = pd.DataFrame(rows)
    base = df[df.yaw_deg == 0].set_index("frame").score
    df["ratio"] = df.score / df.frame.map(base)
    s0 = base.values
    abs_thr = float(s0.mean() - 2 * s0.std())
    df["detect_ratio"] = df.ratio < args.ratio_thr
    df["detect_abs"] = df.score < abs_thr
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)

    summ = df.groupby("yaw_deg").agg(score_mean=("score", "mean"), ratio_mean=("ratio", "mean"),
                                     detect_ratio=("detect_ratio", "mean"), detect_abs=("detect_abs", "mean")).round(3)
    print(f"abs threshold (mean-2std @ yaw0) = {abs_thr:.3f}; ratio threshold = {args.ratio_thr}")
    print(summ.to_string())
    first = df[df.detect_ratio].groupby("frame").yaw_deg.min().reindex(args.frames)
    print("yaw nhỏ nhất bị phát hiện (ratio) theo frame:\n" + first.to_string())
    print(f"-> {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for fid, g in df.groupby("frame"):
        ax[0].plot(g.yaw_deg, g.score, alpha=0.6, label=fid)
    ax[0].axhline(abs_thr, color="k", ls="--", label=f"ngưỡng tuyệt đối {abs_thr:.2f}")
    ax[0].set(xlabel="yaw drift (deg)", ylabel="alignment score", title="Score theo yaw (mỗi đường = 1 frame)")
    ax[0].legend(fontsize=6, ncol=2), ax[0].grid(alpha=0.3)
    ax[1].plot(summ.index, summ.detect_ratio * 100, "o-", label=f"ratio < {args.ratio_thr}")
    ax[1].plot(summ.index, summ.detect_abs * 100, "s--", label="ngưỡng tuyệt đối")
    ax[1].set(xlabel="yaw drift (deg)", ylabel="% frame phát hiện drift", title="Tỉ lệ phát hiện"), ax[1].legend(), ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig_path = out.parent / "figures" / "alignment_score_vs_yaw.png"
    fig.savefig(fig_path, dpi=130)
    print(f"-> {fig_path}")


if __name__ == "__main__":
    main()
