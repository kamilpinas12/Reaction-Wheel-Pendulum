#!/usr/bin/env python3
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_ip_ml(md_path: Path):
    text = md_path.read_text(encoding="utf-8")

    # Capture sections that look like "## <label>" followed by an Ip/ml line.
    pattern = re.compile(
        r"^##\s+(?P<label>.+?)\s*$[\s\S]*?^Ip:\s*(?P<Ip>[+-]?[0-9]*\.?[0-9]+)\s*,\s*f:\s*[+-]?[0-9]*\.?[0-9]+\s*,\s*ml:\s*(?P<ml>[+-]?[0-9]*\.?[0-9]+)",
        re.MULTILINE,
    )

    points = []
    for match in pattern.finditer(text):
        label = match.group("label").strip()
        ip = float(match.group("Ip"))
        ml = float(match.group("ml"))
        points.append((label, ip, ml))

    return points


def main():
    md_path = Path(__file__).with_name("model_parameters.md")
    points = parse_ip_ml(md_path)

    if not points:
        raise SystemExit("No Ip/ml pairs found in model_parameters.md")

    labels = [p[0] for p in points]
    ips = [p[1] for p in points]
    mls = [p[2] for p in points]

    # Correlation line: ml = -4.52 * Ip + 0.14
    ip_min, ip_max = min(ips), max(ips)
    ip_line = [ip_min, ip_max]
    ml_line = [-4.52 * ip + 0.14 for ip in ip_line]

    # Best-fit linear regression (least squares)
    ips_arr = np.array(ips, dtype=float)
    mls_arr = np.array(mls, dtype=float)
    slope, intercept = np.polyfit(ips_arr, mls_arr, 1)
    reg_line = [slope * ip + intercept for ip in ip_line]

    # Compute R^2 for the fit
    pred_mls = slope * ips_arr + intercept
    ss_res = np.sum((mls_arr - pred_mls) ** 2)
    ss_tot = np.sum((mls_arr - np.mean(mls_arr)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    plt.figure(figsize=(8, 6))
    plt.scatter(ips, mls, color="tab:blue", zorder=3)
    plt.plot(ip_line, ml_line, color="tab:orange", linewidth=2, label="ml = -4.52 * Ip + 0.14")
    plt.plot(ip_line, reg_line, color="tab:green", linestyle="--", linewidth=2,
             label=f"Best fit: ml = {slope:.4f} * Ip + {intercept:.4f} (R²={r2:.4f})")

    for label, ip, ml in points:
        plt.annotate(label, (ip, ml), textcoords="offset points", xytext=(6, 6), fontsize=9)

    plt.xlabel("Ip")
    plt.ylabel("ml")
    plt.title("Ip vs ml with correlation line")
    plt.grid(True, linestyle="--", alpha=0.4, zorder=0)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
