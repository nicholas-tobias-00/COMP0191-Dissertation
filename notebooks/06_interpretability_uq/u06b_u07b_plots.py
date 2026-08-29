"""U-06b/U-07b figures: reuses `u06_cqr_comparison_plots.run()` and
`u07_lsu_cqr_comparison_plot.run()` UNCHANGED (both already take chains/summary filenames as
parameters), pointed at the B18-derived chains/summaries instead of U04/U05's. Same output
directories as U-06/U-07 (`figures/u06_cqr`, `figures/u07_lsu_cqr`) -- filenames already encode
`data_label`, so nothing collides with the existing U04/U05 figures, matching how those two
labels already coexist in the same directories.
"""
import sys

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\notebooks\06_interpretability_uq")

import u06_cqr_comparison_plots as u06p
import u07_lsu_cqr_comparison_plot as u07p


def main():
    u06p.run("U08b18", "forecast_daily_v3.csv", "u08_chains.csv", "u08_summary.csv",
              "u06b_u08_cqr_summary.csv", [2, 4, 9], [2018, 2019, 2020, 2021, 2022])
    u06p.run("U05b18", "forecast_daily_v3.csv", "u05b_chains.csv", "u05b_summary.csv",
              "u06b_u05b_cqr_summary.csv", [2, 4, 9], [2018, 2019, 2020, 2021, 2022])

    u07p.run("U08b18", "u08_chains.csv", "u07b_u08_lsu_cqr_summary.csv", "u06b_u08_cqr_summary.csv",
             towers=[2, 4, 9], models=["B18_TabPFN_champion"])
    u07p.run("U05b18", "u05b_chains.csv", "u07b_u05b_lsu_cqr_summary.csv", "u06b_u05b_cqr_summary.csv",
             towers=[2, 4, 9], models=["Direct_TabICLv2_solo_trend"])


if __name__ == "__main__":
    main()
