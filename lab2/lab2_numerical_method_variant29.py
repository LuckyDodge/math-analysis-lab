from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt


L = 3.0 * math.pi
P = math.pi / 2.0
PLOT_NS = [3, 10, 30]
SAMPLE_POINTS = 4000


def f_on_interval(x: float) -> float:
    if 0.0 <= x < P:
        return math.sin(x)
    if P <= x <= L:
        return 1.0
    raise ValueError("x must belong to [0, 3*pi]")


def integral_f() -> float:
    return 1.0 + 5.0 * math.pi / 2.0


def base_a0() -> float:
    return 2.0 * integral_f() / L


def general_coefficients(n: int) -> tuple[float, float]:
    # Coefficients for the 3*pi-periodic extension of f on [0, 3*pi].
    alpha = 2.0 * n / 3.0
    theta = n * math.pi / 3.0
    denominator = alpha * (1.0 - alpha * alpha)

    a_n = (2.0 / L) * (alpha - math.sin(theta)) / denominator
    b_n = (2.0 / L) * (math.cos(theta) + alpha * alpha - 1.0) / denominator
    return a_n, b_n


def cosine_coefficient(n: int) -> float:
    alpha = n / 3.0
    # For n = 3 the closed formula has a removable singularity.
    if n == 3:
        return -1.0 / (3.0 * math.pi)

    theta = n * math.pi / 6.0
    return (2.0 / L) * (alpha - math.sin(theta)) / (alpha * (1.0 - alpha * alpha))


def sine_coefficient(n: int) -> float:
    alpha = n / 3.0
    # The n = 3 coefficient is computed from the integral directly.
    if n == 3:
        return 1.0 / 6.0 + 2.0 / (3.0 * math.pi)

    theta = n * math.pi / 6.0
    numerator = math.cos(theta) + ((-1) ** n) * (alpha * alpha - 1.0)
    return (2.0 / L) * numerator / (alpha * (1.0 - alpha * alpha))


def partial_sum_general(x: float, n_terms: int) -> float:
    total = base_a0() / 2.0
    for n in range(1, n_terms + 1):
        a_n, b_n = general_coefficients(n)
        total += a_n * math.cos(2.0 * n * x / 3.0) + b_n * math.sin(2.0 * n * x / 3.0)
    return total


def partial_sum_cosine(x: float, n_terms: int) -> float:
    total = base_a0() / 2.0
    for n in range(1, n_terms + 1):
        total += cosine_coefficient(n) * math.cos(n * x / 3.0)
    return total


def partial_sum_sine(x: float, n_terms: int) -> float:
    total = 0.0
    for n in range(1, n_terms + 1):
        total += sine_coefficient(n) * math.sin(n * x / 3.0)
    return total


def periodic_general_sum(x: float) -> float:
    r = x % L
    # At jump points the Fourier series converges to the average of one-sided limits.
    if math.isclose(r, 0.0, abs_tol=1e-12) or math.isclose(r, L, abs_tol=1e-12):
        return 0.5
    return math.sin(r) if r < P else 1.0


def periodic_cosine_sum(x: float) -> float:
    t = x % (2.0 * L)
    r = t if t <= L else 2.0 * L - t
    return math.sin(r) if r < P else 1.0


def periodic_sine_sum(x: float) -> float:
    t = ((x + L) % (2.0 * L)) - L
    if math.isclose(t, 0.0, abs_tol=1e-12) or math.isclose(abs(t), L, abs_tol=1e-12):
        return 0.0

    sign = 1.0 if t > 0 else -1.0
    r = abs(t)
    value = math.sin(r) if r < P else 1.0
    return sign * value


def build_x_values(left: float, right: float) -> list[float]:
    return [left + (right - left) * i / SAMPLE_POINTS for i in range(SAMPLE_POINTS + 1)]


def save_series_plot(
    output_dir: Path,
    filename: str,
    title: str,
    exact_sum: Callable[[float], float],
    partial_sum: Callable[[float, int], float],
    left: float,
    right: float,
) -> Path:
    x_values = build_x_values(left, right)

    plt.figure(figsize=(11, 6))
    # Draw the theoretical sum together with several partial sums.
    plt.plot(
        x_values,
        [exact_sum(x) for x in x_values],
        color="black",
        linewidth=2.0,
        label="sum of the Fourier series",
    )

    for n_terms in PLOT_NS:
        start = time.perf_counter()
        y_values = [partial_sum(x, n_terms) for x in x_values]
        elapsed = time.perf_counter() - start
        plt.plot(
            x_values,
            y_values,
            linewidth=1.1,
            label=f"N = {n_terms}, time = {elapsed:.3f}s",
        )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = output_dir / filename
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def max_error_away_from_jumps(
    exact_sum: Callable[[float], float],
    partial_sum: Callable[[float, int], float],
    n_terms: int,
    left: float,
    right: float,
    excluded_points: list[float],
    radius: float = 0.06,
) -> float:
    max_error = 0.0
    for x in build_x_values(left, right):
        if any(abs(x - point) < radius for point in excluded_points):
            continue
        max_error = max(max_error, abs(partial_sum(x, n_terms) - exact_sum(x)))
    return max_error


def format_table(headers: list[str], rows: list[list[object]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    def render(row: list[object]) -> str:
        return " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "numerical_method_outputs"
    output_dir.mkdir(exist_ok=True)

    plots = [
        save_series_plot(
            output_dir,
            "variant29_general_fourier.png",
            "General trigonometric Fourier series, period 3*pi",
            periodic_general_sum,
            partial_sum_general,
            -L,
            2.0 * L,
        ),
        save_series_plot(
            output_dir,
            "variant29_cosine_fourier.png",
            "Cosine Fourier series for the even extension, period 6*pi",
            periodic_cosine_sum,
            partial_sum_cosine,
            -L,
            L,
        ),
        save_series_plot(
            output_dir,
            "variant29_sine_fourier.png",
            "Sine Fourier series for the odd extension, period 6*pi",
            periodic_sine_sum,
            partial_sum_sine,
            -L,
            L,
        ),
    ]

    rows: list[list[object]] = []
    series_data = [
        (
            "general",
            periodic_general_sum,
            partial_sum_general,
            -L,
            2.0 * L,
            [-L, 0.0, L, 2.0 * L],
        ),
        (
            "cosine",
            periodic_cosine_sum,
            partial_sum_cosine,
            -L,
            L,
            [],
        ),
        (
            "sine",
            periodic_sine_sum,
            partial_sum_sine,
            -L,
            L,
            [-L, L],
        ),
    ]

    for name, exact, partial, left, right, jumps in series_data:
        for n_terms in PLOT_NS:
            error = max_error_away_from_jumps(exact, partial, n_terms, left, right, jumps)
            rows.append([name, n_terms, f"{error:.6f}"])

    report_lines = [
        "Numerical method for Lab 4-2, variant 29",
        "",
        "Function on [0, 3*pi]:",
        "f(x) = sin(x), 0 <= x < pi/2",
        "f(x) = 1,      pi/2 <= x <= 3*pi",
        "",
        "Partial sums are plotted for N = 3, 10, 30.",
        "",
        "Maximum errors on the plotting interval, away from jump points:",
        format_table(["series", "N", "max abs error"], rows),
        "",
        "Plots:",
        *(str(path) for path in plots),
    ]

    report_path = output_dir / "variant29_numerical_results.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
