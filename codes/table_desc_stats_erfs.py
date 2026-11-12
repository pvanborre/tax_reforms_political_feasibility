import numpy as np
import pandas as pd
import os

# Years covered by the ERFS surveys
YEARS = list(range(2002, 2020))


def weighted_percentile(data, weights, perc):
    """
    Computes a weighted percentile.
    perc is in [0, 100].
    """
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]
    threshold = perc / 100.0 * total_weight
    idx = np.searchsorted(cumulative_weights, threshold)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def compute_stats(year):
    """
    Reads the CSV for a given ERFS year and computes all statistics.
    Returns None if the file does not exist.
    """
    beginning_year = year
    end_year = year + 1
    filepath = f"excel/{beginning_year}-{end_year}/people_adults_{beginning_year}-{end_year}.csv"

    if not os.path.exists(filepath):
        print(f"Warning: file not found for year {beginning_year}, skipping.")
        return None

    df = pd.read_csv(filepath)
    total_earning = df["total_earning"].values
    weights = df["wprm"].values

    n = len(df)
    n_weighted = weights.sum()
    mean = np.average(total_earning, weights=weights)
    std = np.sqrt(np.average((total_earning - mean) ** 2, weights=weights))
    p10 = weighted_percentile(total_earning, weights, 10)
    p25 = weighted_percentile(total_earning, weights, 25)
    p50 = weighted_percentile(total_earning, weights, 50)
    p75 = weighted_percentile(total_earning, weights, 75)
    p90 = weighted_percentile(total_earning, weights, 90)
    p99 = weighted_percentile(total_earning, weights, 99)

    return {
        "Year": beginning_year,
        "N": n,
        "N weighted (M)": round(n_weighted / 1e6, 2),
        "Mean": round(mean, 0),
        "Std": round(std, 0),
        "P10": round(p10, 0),
        "P25": round(p25, 0),
        "P50": round(p50, 0),
        "P75": round(p75, 0),
        "P90": round(p90, 0),
        "P99": round(p99, 0),
    }


def make_latex_table(df_stats):
    """
    Produces a LaTeX table from the stats dataframe.
    """
    col_format = "l" + "r" * (len(df_stats.columns) - 1)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{ERFS sample size and income distribution}")
    lines.append(r"\label{tab:erfs_distribution}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + col_format + r"}")
    lines.append(r"\toprule")

    # header
    header = " & ".join(df_stats.columns.tolist()) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # rows
    for _, row in df_stats.iterrows():
        formatted = []
        for col, val in row.items():
            if col == "Year":
                formatted.append(str(int(val)))
            elif col == "N":
                formatted.append(f"{int(val):,}")
            elif col == "N weighted (M)":
                formatted.append(f"{val:.2f}")
            else:
                formatted.append(f"{int(val):,}")
        lines.append(" & ".join(formatted) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{\linewidth}")
    lines.append(r"\smallskip")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes:} Sample of adults (age $\geq$ 18) from the ERFS-FPR surveys.")
    lines.append(r"All income statistics are weighted by survey weights \textit{wprm} and expressed in euros.")
    lines.append(r"Total income excludes capital income.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    rows = []
    for year in YEARS:
        stats = compute_stats(year)
        if stats is not None:
            rows.append(stats)

    df_stats = pd.DataFrame(rows)
    print(df_stats.to_string(index=False))

    latex_table = make_latex_table(df_stats)

    output_path = "../outputs/table_erfs.tex"
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {output_path}")


main()
