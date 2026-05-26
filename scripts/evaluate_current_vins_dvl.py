#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path


ROOT = Path("/Users/kanghyunmin/Desktop/under_water_image_match")
ROSBAG_INPUT_DIR = ROOT / "data/rosbag_active"
DVL = ROSBAG_INPUT_DIR / "dvl_reference_0_30s.csv"
VINS = ROOT / "outputs/ros2_vins_fusion_live/vio.csv"
LOG = Path("/tmp/uuv_rviz_run/vins_node.log")


def read_xyz(path: Path, flip_y: bool = False):
    if not path.exists():
        return []
    lines = path.read_text(errors="ignore").splitlines()
    if not lines:
        return []

    rows = []
    if any(c.isalpha() for c in lines[0]):
        from io import StringIO

        reader = csv.DictReader(StringIO("\n".join(lines)))
        for row in reader:
            keys = {k.lower(): k for k in row.keys() if k}

            def get(*names):
                for name in names:
                    if name in keys and row[keys[name]] != "":
                        return float(row[keys[name]])
                raise KeyError(names)

            try:
                x = get("x", "px", "p_x")
                y = get("y", "py", "p_y")
                z = get("z", "pz", "p_z")
            except Exception:
                continue
            rows.append((x, -y if flip_y else y, z))
    else:
        for line in lines:
            vals = []
            for item in line.split(","):
                try:
                    vals.append(float(item))
                except ValueError:
                    pass
            if len(vals) >= 4:
                x, y, z = vals[1], vals[2], vals[3]
                rows.append((x, -y if flip_y else y, z))
    return rows


def path_length(points):
    return sum(math.dist(points[i - 1], points[i]) for i in range(1, len(points)))


def resample(points, count):
    if not points or count <= 0:
        return []
    if len(points) == 1:
        return [points[0]] * count

    distances = [0.0]
    for i in range(1, len(points)):
        distances.append(distances[-1] + math.dist(points[i - 1], points[i]))

    total = distances[-1]
    if total <= 1e-12:
        return [points[0]] * count

    out = []
    j = 1
    for k in range(count):
        target = total * k / (count - 1) if count > 1 else 0.0
        while j < len(distances) - 1 and distances[j] < target:
            j += 1
        a, b = distances[j - 1], distances[j]
        u = 0.0 if b == a else (target - a) / (b - a)
        out.append(tuple(points[j - 1][m] * (1.0 - u) + points[j][m] * u for m in range(3)))
    return out


def zero_start(points):
    if not points:
        return []
    origin = points[0]
    return [(x - origin[0], y - origin[1], z - origin[2]) for x, y, z in points]


def corr(a, b):
    n = len(a)
    if n < 2:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 1e-12 or vb <= 1e-12:
        return None
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


def main():
    dvl = read_xyz(DVL, flip_y=True)
    vins = read_xyz(VINS, flip_y=False)
    print("DVL n", len(dvl), "length", path_length(dvl), "end", dvl[-1] if dvl else None)
    print("VINS n", len(vins), "length", path_length(vins), "end", vins[-1] if vins else None)

    n = min(len(dvl), len(vins), 400)
    if n > 2:
        d = zero_start(resample(dvl, n))
        v = zero_start(resample(vins, n))
        errors = [math.dist(d[i], v[i]) for i in range(n)]
        out = {
            "n": n,
            "rmse3": math.sqrt(sum(e * e for e in errors) / n),
            "mean_err": sum(errors) / n,
            "max_err": max(errors),
            "dvl_length": path_length(d),
            "vins_length": path_length(v),
            "axis": {},
        }
        for axis, name in enumerate("xyz"):
            da = [p[axis] for p in d]
            va = [p[axis] for p in v]
            out["axis"][name] = {
                "rmse": math.sqrt(sum((da[i] - va[i]) ** 2 for i in range(n)) / n),
                "corr": corr(da, va),
                "v_range": [min(va), max(va)],
                "d_range": [min(da), max(da)],
            }
        print(json.dumps(out, indent=2))
    else:
        print("not enough samples for metrics")

    log = LOG.read_text(errors="ignore") if LOG.exists() else ""
    for pattern, name in [
        ("throw img0", "throw_img0"),
        ("throw img1", "throw_img1"),
        ("underwater pnp rejected", "reject_count"),
        ("unstable tracking", "unstable_count"),
        ("feature tracking not enough", "not_enough_count"),
    ]:
        print(name, log.count(pattern))
    print("\n".join(log.splitlines()[-25:]))


if __name__ == "__main__":
    main()
