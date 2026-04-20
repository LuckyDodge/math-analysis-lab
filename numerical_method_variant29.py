from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt


E_LEFT = 0.0
E_RIGHT = 4.0
PLOT_NS = [1, 2, 5, 10]  # small n for plots (dyadic scheme)
LEBESGUE_NS = [10, 100, 1000]  # large n for Lebesgue integral
STIELTJES_NS = [50, 500, 5000]  # large n for Stieltjes integral


def f(x: float) -> float:
    return math.exp(x)


def simple_fn_dyadic(x: float, n: int) -> float:
    """
    Dyadic simple approximation: fn = floor(2^n * e^x) / 2^n.
    Used for plots and theoretical justification.
    """
    scale = 2 ** n
    return math.floor(scale * math.exp(x)) / scale


def simple_fn_numeric(x: float, n: int) -> float:
    """
    Numeric simple approximation: fn = floor(n * e^x) / n.
    Used for computing integrals for large n.
    """
    return math.floor(n * math.exp(x)) / n


def analytic_lebesgue_integral() -> float:
    return math.exp(E_RIGHT) - math.exp(E_LEFT)


def analytic_stieltjes_integral() -> float:
    return sum(math.exp(math.sqrt(k)) for k in range(17))


def lebesgue_integral_of_simple_fn(n: int) -> float:
    """
    Exact integral of floor(n e^x) / n over [0, 4].
    On each interval [ln(k/n), ln((k+1)/n)) the function equals k/n.
    """
    max_k = math.floor(n * math.exp(E_RIGHT))
    total = 0.0

    for k in range(n, max_k):
        left = math.log(k / n)
        right = math.log((k + 1) / n)
        total += (k / n) * (right - left)

    last_left = math.log(max_k / n)
    total += (max_k / n) * (E_RIGHT - last_left)
    return total


def stieltjes_integral_of_simple_fn(n: int) -> float:
    """
    Exact integral with respect to mu_F for F(x) = ceil(x^2).
    The measure is atomic on points sqrt(k), k = 0, ..., 16, each with mass 1.
    """
    return sum(simple_fn_numeric(math.sqrt(k), n) for k in range(17))


def build_plot(output_dir: Path) -> Path:
    x_values = [E_LEFT + (E_RIGHT - E_LEFT) * i / 2000 for i in range(2001)]
    y_values = [f(x) for x in x_values]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="f(x) = exp(x)", linewidth=2.0, color="black")

    for n in PLOT_NS:
        y_simple = [simple_fn_dyadic(x, n) for x in x_values]
        plt.step(x_values, y_simple, where="post", label=f"f_n (dyadic), n={n}", linewidth=1.2)

    plt.title("Lower simple approximations for f(x) = exp(x) on [0, 4]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "variant29_fn_plots.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def format_table(headers: list[str], rows: list[list[object]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    def render_row(row: list[object]) -> str:
        return " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "numerical_method_outputs"
    output_dir.mkdir(exist_ok=True)

    lebesgue_exact = analytic_lebesgue_integral()
    stieltjes_exact = analytic_stieltjes_integral()

    # Compute Lebesgue integral
    lebesgue_rows: list[list[object]] = []
    for n in LEBESGUE_NS:
        start = time.perf_counter()
        value = lebesgue_integral_of_simple_fn(n)
        elapsed = time.perf_counter() - start
        error = abs(value - lebesgue_exact)
        lebesgue_rows.append(
            [
                n,
                f"{value:.12f}",
                f"{lebesgue_exact:.12f}",
                f"{error:.12f}",
                f"{elapsed:.6f}",
            ]
        )

    # Compute Stieltjes integral
    stieltjes_rows: list[list[object]] = []
    for n in STIELTJES_NS:
        start = time.perf_counter()
        value = stieltjes_integral_of_simple_fn(n)
        elapsed = time.perf_counter() - start
        error = abs(value - stieltjes_exact)
        stieltjes_rows.append(
            [
                n,
                f"{value:.12f}",
                f"{stieltjes_exact:.12f}",
                f"{error:.12f}",
                f"{elapsed:.6f}",
            ]
        )

    # Build plots
    plot_path = build_plot(output_dir)

    # Generate report
    report_lines = [
        "Numerical method for Lab 4-1, variant 29",
        "",
        "Function: f(x) = exp(x)",
        "Set: E = [0, 4]",
        "Stieltjes generator: F(x) = ceil(x^2)",
        "",
        "Simple approximations used:",
        "1. Dyadic scheme (theory & plots): f_n(x) = floor(2^n * exp(x)) / 2^n",
        "2. Numeric scheme (computations): g_n(x) = floor(n * exp(x)) / n",
        "",
        f"Analytic Lebesgue integral: {lebesgue_exact:.12f}",
        f"Analytic Lebesgue-Stieltjes integral: {stieltjes_exact:.12f}",
        "",
        "Lebesgue integral of g_n on [0, 4]",
        format_table(
            ["n", "numeric value", "analytic value", "abs error", "time (s)"],
            lebesgue_rows,
        ),
        "",
        "Lebesgue-Stieltjes integral of g_n on [0, 4]",
        format_table(
            ["n", "numeric value", "analytic value", "abs error", "time (s)"],
            stieltjes_rows,
        ),
        "",
        f"Plot saved to: {plot_path}",
    ]

    report_path = output_dir / "variant29_numerical_results.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()