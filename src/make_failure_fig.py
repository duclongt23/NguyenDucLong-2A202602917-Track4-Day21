"""Topic A - ảnh failure case: calib gốc | calib lệch, crop quanh 2D box của một object.

Điểm của object (nằm trong 3D box GT, xác định bằng calib gốc) tô màu magenta, các điểm LiDAR còn lại tô theo depth.

    python src/make_failure_fig.py --frame 000025 --obj-idx 4 --yaw-deg 1 --out results/figures/fail_01_yaw1deg_thin_cyclist_44m.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calib_sweep import points_in_box3d  # noqa: E402
from starter.datasets import load_frame  # noqa: E402
from starter.projection import draw_box2d, overlay_points, perturb_extrinsic, project_velo_to_image  # noqa: E402


def panel(fr, calib, obj, obj_pts, title, margin, scale):
    img = fr["image"]
    uv, depth, _ = project_velo_to_image(fr["points"], calib, img.shape)
    vis = overlay_points(img, uv, depth, radius=1)
    uo, _, _ = project_velo_to_image(obj_pts, calib, img.shape)
    x1, y1, x2, y2 = obj.bbox
    inbox = (uo[:, 0] >= x1) & (uo[:, 0] <= x2) & (uo[:, 1] >= y1) & (uo[:, 1] <= y2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) / 2 + margin
    h, w = img.shape[:2]
    X1, Y1 = int(max(0, cx - half * 1.6)), int(max(0, cy - half))
    X2, Y2 = int(min(w, cx + half * 1.6)), int(min(h, cy + half))
    crop = cv2.resize(vis[Y1:Y2, X1:X2], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for (u, v), ok in zip(uo, inbox):
        col = (0, 200, 0) if ok else (255, 0, 255)
        cv2.circle(crop, (int((u - X1) * scale), int((v - Y1) * scale)), 4, col, -1)
        cv2.circle(crop, (int((u - X1) * scale), int((v - Y1) * scale)), 4, (255, 255, 255), 1)
    cv2.rectangle(crop, (int((x1 - X1) * scale), int((y1 - Y1) * scale)),
                  (int((x2 - X1) * scale), int((y2 - Y1) * scale)), (0, 255, 0), 2)
    pct = 100 * inbox.mean() if len(inbox) else float("nan")
    bar = np.full((70, crop.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(bar, f"{pct:.0f}% object points in 2D box ({int(inbox.sum())}/{len(inbox)})", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if pct > 50 else (80, 80, 255), 2)
    return np.vstack([bar, crop])


def main() -> None:
    ap = argparse.ArgumentParser(description="Ảnh failure: object trước/sau khi calib bị lệch")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", required=True)
    ap.add_argument("--obj-idx", type=int, required=True, help="chỉ số object trong label (cột obj_idx của *_per_object.csv)")
    ap.add_argument("--yaw-deg", type=float, default=0.0)
    ap.add_argument("--tx", type=float, default=0.0, help="mét")
    ap.add_argument("--ty", type=float, default=0.0, help="mét")
    ap.add_argument("--tz", type=float, default=0.0, help="mét")
    ap.add_argument("--margin", type=float, default=25, help="pixel quanh box khi crop")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fr = load_frame(args.data_root, args.frame)
    obj = fr["labels"][args.obj_idx]
    pts = fr["points"][:, :3].astype(np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    fr["points"] = pts
    obj_pts = pts[points_in_box3d(pts, fr["calib"], obj)]
    bad = perturb_extrinsic(fr["calib"], yaw_deg=args.yaw_deg, t_xyz_m=(args.tx, args.ty, args.tz))
    name = f"{obj.type} @ {obj.location[2]:.0f} m"
    a = panel(fr, fr["calib"], obj, obj_pts, f"{name}: calib goc", args.margin, args.scale)
    drift = f"yaw {args.yaw_deg}deg" if args.yaw_deg else f"t=({args.tx},{args.ty},{args.tz}) m"
    b = panel(fr, bad, obj, obj_pts, f"{name}: lech {drift}", args.margin, args.scale)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack([a, b]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
