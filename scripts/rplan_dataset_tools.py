#!/usr/bin/env python3
"""
RPLAN 本机数据包工具：盘点 Network/*.mat、外轮廓几何质检、从 mat 重建 DeepLayout 所需的四通道语义 PNG。

双用途（对应计划中的两条路径）：
- geom-qc：不依赖四通道图，直接从 boundary 折线做轮廓角点等检查，供 Holodeck / 过滤名单使用。
- export-deeplayout：从 Graph2Plan 文档所述字段（boundary / rBoundary / rType / order）光栅化到
  256×256×4 uint8，与 write_pickle.py 通道约定一致；本仓库 rType 仅有 0–12，内墙/内门通道在图中恒为 0，
  与当前 data_train.mat 一致，不等价于作者原始栅格金标。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import scipy.io as sio
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要 scipy：conda/pip install scipy") from e

# 默认数据根（相对本仓库）
DEFAULT_RPLAN_ROOT = Path(__file__).resolve().parent.parent / "RPLAN dataset"


def load_mat_samples(mat_path: Path):
    raw = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "data" not in raw:
        raise KeyError(f"{mat_path} 中无 'data' 变量，键为：{list(raw.keys())}")
    return raw["data"]


def inventory(rplan_root: Path) -> None:
    net = rplan_root / "Network"
    candidates = [
        net / "data.mat",
        net / "data" / "data_train.mat",
        net / "data" / "data_valid.mat",
        net / "data" / "data_test.mat",
    ]
    print("RPLAN 根目录:", rplan_root.resolve())
    for p in candidates:
        if not p.exists():
            print(f"  [缺失] {p}")
            continue
        data = load_mat_samples(p)
        first = data.flat[0]
        fields = list(first._fieldnames) if hasattr(first, "_fieldnames") else []
        print(f"  [OK] {p} 样本数={len(data)} 首条字段={fields}")


def _vertex_angles_deg(xy: np.ndarray) -> np.ndarray:
    """闭合折线顶点处两邻边夹角（度），范围 [0, 180]。"""
    n = len(xy)
    if n < 3:
        return np.array([])
    angles = []
    for i in range(n):
        a = xy[i - 1].astype(np.float64) - xy[i]
        b = xy[(i + 1) % n].astype(np.float64) - xy[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        angles.append(np.degrees(np.arccos(c)))
    return np.asarray(angles, dtype=np.float64)


def geom_qc(mat_path: Path, out_csv: Path | None, limit: int | None, acute_deg: float) -> None:
    data = load_mat_samples(mat_path)
    n = len(data) if limit is None else min(len(data), limit)
    rows = []
    acute_count = 0
    for i in range(n):
        s = data[i]
        xy = np.asarray(s.boundary[:, :2], dtype=np.int32)
        ang = _vertex_angles_deg(xy)
        amin = float(ang.min()) if ang.size else float("nan")
        flag = int(amin < acute_deg) if ang.size else 0
        acute_count += flag
        rows.append({"name": s.name, "n_boundary": len(xy), "min_angle_deg": round(amin, 3), "acute_lt_thresh": flag})
    print(f"geom-qc: 文件={mat_path} 检查条数={n} 最小角<{acute_deg}° 的样本数={acute_count}")
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
            w.writeheader()
            w.writerows(rows)
        print("  已写 CSV:", out_csv)


def _line_mask(shape_hw, x0, y0, x1, y1, width: int, value: int, canvas: np.ndarray) -> None:
    img = Image.fromarray(canvas, mode="L")
    d = ImageDraw.Draw(img)
    d.line([(int(x0), int(y0)), (int(x1), int(y1))], fill=int(value), width=int(width))
    canvas[:] = np.asarray(img)


def _poly_mask(shape_hw, pts_xy: np.ndarray, fill: int) -> np.ndarray:
    h, w = shape_hw
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    loop = [(int(p[0]), int(p[1])) for p in pts_xy]
    if len(loop) >= 3:
        d.polygon(loop, fill=int(fill))
    return np.asarray(img, dtype=np.uint8)


def mat_struct_to_semantic_rgba(s) -> np.ndarray:
    """
    返回 uint8 数组 (H,W,4): boundary, category, index, inside。
    前门：boundary 通道上首条边 boundary[0]-boundary[1] 画为 255（与 Graph2Plan README 一致）。
    """
    h = w = 256
    boundary = np.zeros((h, w), dtype=np.uint8)
    category = np.zeros((h, w), dtype=np.uint8)
    index = np.zeros((h, w), dtype=np.uint8)
    inside = np.zeros((h, w), dtype=np.uint8)

    b = np.asarray(s.boundary[:, :2], dtype=np.float32)
    n = len(b)
    if n < 3:
        return np.stack([boundary, category, index, inside], axis=-1)

    # 外轮廓填充 -> inside
    inside[:] = _poly_mask((h, w), b, 255)

    # 房间：按 order 升序绘制，后画覆盖先画；再画墙线，避免房间盖住外墙
    order_idx = np.argsort(np.asarray(s.order), kind="mergesort")
    rtype = np.asarray(s.rType)
    for k in order_idx:
        pts = np.asarray(s.rBoundary[k], dtype=np.float32)
        if pts.shape[0] < 3:
            continue
        m = _poly_mask((h, w), pts, 1).astype(bool)
        rt = int(rtype[k])
        if 0 <= rt <= 12:
            category[m] = np.uint8(rt)
        index[m] = np.uint8(k + 1)

    # 墙线：除「前门首边」外画 127；首边画 255
    for i in range(n):
        x0, y0 = b[i]
        x1, y1 = b[(i + 1) % n]
        if i == 0:
            _line_mask((h, w), x0, y0, x1, y1, width=4, value=255, canvas=boundary)
        else:
            _line_mask((h, w), x0, y0, x1, y1, width=2, value=127, canvas=boundary)

    return np.stack([boundary, category, index, inside], axis=-1)


def export_deeplayout(mat_path: Path, out_dir: Path, limit: int | None, skip_existing: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_mat_samples(mat_path)
    n = len(data) if limit is None else min(len(data), limit)
    wrote = 0
    for i in range(n):
        s = data[i]
        name = str(s.name)
        dst = out_dir / f"{name}.png"
        if skip_existing and dst.exists():
            continue
        rgba = mat_struct_to_semantic_rgba(s)
        Image.fromarray(rgba, mode="RGBA").save(dst)
        wrote += 1
    print(f"export-deeplayout: 自 {mat_path} 写出 {wrote} 张 PNG 到 {out_dir}（处理前 {n} 条）")
    return wrote


def main(argv=None):
    p = argparse.ArgumentParser(description="RPLAN mat 工具：inventory / geom-qc / export-deeplayout")
    p.add_argument("--rplan-root", type=Path, default=DEFAULT_RPLAN_ROOT, help="解压后的 RPLAN dataset 根目录")
    sub = p.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("inventory", help="列出 Network 下各 mat 与字段")
    p0.set_defaults(func=lambda a: inventory(a.rplan_root))

    p1 = sub.add_parser("geom-qc", help="外轮廓折线顶点最小夹角统计（用于过滤锐角轮廓）")
    p1.add_argument("--mat", type=Path, default=None, help="默认 data/data_train.mat")
    p1.add_argument("--out-csv", type=Path, default=None)
    p1.add_argument("--limit", type=int, default=None)
    p1.add_argument("--acute-deg", type=float, default=90.0, help="最小角低于该值则标记 acute_lt_thresh=1")

    def _geom(a):
        mp = a.mat or (a.rplan_root / "Network" / "data" / "data_train.mat")
        geom_qc(mp, a.out_csv, a.limit, a.acute_deg)

    p1.set_defaults(func=_geom)

    p2 = sub.add_parser("export-deeplayout", help="从 mat 生成四通道 PNG（DeepLayout write_pickle 输入）")
    p2.add_argument("--mat", type=Path, required=True)
    p2.add_argument("--out-dir", type=Path, required=True)
    p2.add_argument("--limit", type=int, default=None)
    p2.add_argument("--skip-existing", action="store_true")

    def _exp(a):
        export_deeplayout(a.mat, a.out_dir, a.limit, a.skip_existing)

    p2.set_defaults(func=_exp)

    a = p.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
