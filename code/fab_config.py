"""
Single source of truth for the simulated Technology Development (TD) fab network.

Everything here is ILLUSTRATIVE: plausible orders of magnitude for a development
fab, not data from any real company. The simulation, the capacity model, the
forecast model and the generated documents all read from this file, so a number
on the dashboard can always be traced back to an assumption listed here.

Units: wafers, lots of 25 wafers, hours, wafer starts per week (WSPW).
"""

LOT_SIZE = 25
HOURS_PER_WEEK = 168

SITES = {
    "FAB_A": "Fab A (development fab with EUV)",
    "FAB_B": "Fab B (development fab, DUV only)",
}

# ---------------------------------------------------------------- tool groups
# min_per_wafer: tool minutes per wafer per pass (single-wafer equivalent).
# Furnaces are batch tools: a 6-hour run holds up to 4 lots; TD lots rarely fill
# a batch, so we assume 3 lots (75 wafers) per run -> 6*60/75 = 4.8 min/wafer.
# E10 time split (fractions of calendar time not available for production):
#   pm = scheduled downtime, down = unscheduled downtime, eng = engineering time
#   (tool used for process development experiments, typical of a TD fab).
# Metrology only measures a sample of lots at each metrology step.
TOOL_GROUPS = {
    "EUV":   {"name": "Litho EUV scanner",      "area": "Litho",      "min_per_wafer": 1.00, "pm": 0.06, "down": 0.09, "eng": 0.15},
    "DUV":   {"name": "Litho DUV immersion",    "area": "Litho",      "min_per_wafer": 0.40, "pm": 0.04, "down": 0.05, "eng": 0.10},
    "ETCH":  {"name": "Dry etch",               "area": "Etch",       "min_per_wafer": 2.00, "pm": 0.05, "down": 0.06, "eng": 0.12},
    "DEP":   {"name": "CVD/ALD deposition",     "area": "Thin films", "min_per_wafer": 2.40, "pm": 0.05, "down": 0.05, "eng": 0.10},
    "PVD":   {"name": "PVD metallization",      "area": "Thin films", "min_per_wafer": 1.50, "pm": 0.04, "down": 0.04, "eng": 0.08},
    "CMP":   {"name": "CMP",                    "area": "Planarization", "min_per_wafer": 2.40, "pm": 0.06, "down": 0.06, "eng": 0.08},
    "WET":   {"name": "Wet clean",              "area": "Wet",        "min_per_wafer": 1.00, "pm": 0.03, "down": 0.04, "eng": 0.05},
    "FURN":  {"name": "Diffusion furnace (batch)", "area": "Diffusion", "min_per_wafer": 4.80, "pm": 0.04, "down": 0.04, "eng": 0.06,
              "batch_lots": 3, "run_hours": 6.0},
    "IMPL":  {"name": "Ion implant",            "area": "Implant",    "min_per_wafer": 1.00, "pm": 0.05, "down": 0.05, "eng": 0.08},
    "METRO": {"name": "Metrology & inspection", "area": "Metrology",  "min_per_wafer": 1.50, "pm": 0.03, "down": 0.03, "eng": 0.10,
              "sample_rate": 0.35},
}
TOOL_GROUP_ORDER = list(TOOL_GROUPS)


def production_availability(tg: str) -> float:
    t = TOOL_GROUPS[tg]
    return 1.0 - t["pm"] - t["down"] - t["eng"]


# ---------------------------------------------------------------- programs
# passes = how many times a wafer visits each tool group over the full route
# (wafers return to the same tool groups layer after layer: re-entrant flow).
# time_mult = process-time multiplier vs. the tool group default (e.g. 3D NAND
# high-aspect-ratio etch takes far longer per pass).
PROGRAMS = {
    "DRAM_A": {"name": "DRAM Node A", "mask_layers": 36, "time_mult": {},
               "passes": {"EUV": 0, "DUV": 36, "ETCH": 45, "DEP": 55, "PVD": 8, "CMP": 14,
                          "WET": 80, "FURN": 12, "IMPL": 18, "METRO": 60}},
    "DRAM_B": {"name": "DRAM Node B (EUV)", "mask_layers": 40, "time_mult": {},
               "passes": {"EUV": 7, "DUV": 33, "ETCH": 52, "DEP": 62, "PVD": 9, "CMP": 16,
                          "WET": 90, "FURN": 13, "IMPL": 20, "METRO": 70}},
    "NAND_C": {"name": "3D NAND Gen C", "mask_layers": 28, "time_mult": {"ETCH": 2.2},
               "passes": {"EUV": 0, "DUV": 28, "ETCH": 40, "DEP": 110, "PVD": 7, "CMP": 12,
                          "WET": 70, "FURN": 10, "IMPL": 10, "METRO": 55}},
    "HBM_TV": {"name": "HBM test vehicle", "mask_layers": 44, "time_mult": {"ETCH": 1.2},
               "passes": {"EUV": 7, "DUV": 37, "ETCH": 60, "DEP": 70, "PVD": 14, "CMP": 22,
                          "WET": 100, "FURN": 13, "IMPL": 20, "METRO": 80}},
    "SHORT": {"name": "Short-loop experiments", "mask_layers": 7, "time_mult": {},
              "passes": {"EUV": 0, "DUV": 7, "ETCH": 10, "DEP": 12, "PVD": 2, "CMP": 3,
                         "WET": 18, "FURN": 3, "IMPL": 4, "METRO": 14}},
}
PROGRAM_ORDER = list(PROGRAMS)

# Which programs are qualified to run at which site (EUV programs need Fab A).
QUALIFIED = {
    "FAB_A": ["DRAM_A", "DRAM_B", "NAND_C", "HBM_TV", "SHORT"],
    "FAB_B": ["DRAM_A", "NAND_C", "SHORT"],
}

HOT_LOT_SHARE = 0.10          # share of lots run at top priority
LINE_YIELD = {"DRAM_A": 0.985, "DRAM_B": 0.970, "NAND_C": 0.980, "HBM_TV": 0.960, "SHORT": 0.990}


def tool_hours_per_wafer(program: str, tg: str) -> float:
    """Tool hours one wafer start consumes at a tool group over the full route."""
    p, t = PROGRAMS[program], TOOL_GROUPS[tg]
    hours = p["passes"][tg] * t["min_per_wafer"] * p["time_mult"].get(tg, 1.0) / 60
    return hours * t.get("sample_rate", 1.0)


# ---------------------------------------------------------------- installed tools
# Installed tool counts today (end of Q3 2026) and planned arrivals.
INSTALLED_TOOLS = {
    "FAB_A": {"EUV": 1, "DUV": 4, "ETCH": 24, "DEP": 36, "PVD": 4, "CMP": 9,
              "WET": 19, "FURN": 13, "IMPL": 5, "METRO": 9},
    "FAB_B": {"EUV": 0, "DUV": 2, "ETCH": 13, "DEP": 19, "PVD": 2, "CMP": 4,
              "WET": 8, "FURN": 6, "IMPL": 2, "METRO": 4},
}
# (site, tool group, quarter the tool is released to production, count)
TOOL_ARRIVALS = [
    ("FAB_A", "EUV", "2027Q4", 1),
    ("FAB_A", "ETCH", "2027Q2", 3),
    ("FAB_B", "DEP", "2027Q3", 3),
]

# ---------------------------------------------------------------- wafer starts
# Actual WSPW in Q3 2026 (drives the 13-week simulation).
ACTUAL_WSPW_2026Q3 = {
    "FAB_A": {"DRAM_A": 250, "DRAM_B": 650, "NAND_C": 150, "HBM_TV": 150, "SHORT": 600},
    "FAB_B": {"DRAM_A": 200, "NAND_C": 350, "SHORT": 400},
}

# Requested WSPW for the 2027 planning horizon (what program teams asked for).
PLAN_QUARTERS = ["2027Q1", "2027Q2", "2027Q3", "2027Q4"]
REQUESTED_WSPW_2027 = {
    "FAB_A": {"DRAM_A": [240, 220, 200, 180], "DRAM_B": [700, 780, 840, 880],
              "NAND_C": [150, 150, 150, 150], "HBM_TV": [170, 210, 250, 270],
              "SHORT": [580, 560, 560, 540]},
    "FAB_B": {"DRAM_A": [190, 180, 170, 160], "NAND_C": [380, 420, 460, 480],
              "SHORT": [380, 380, 380, 380]},
}

# How actual starts have historically compared with requests (used only to
# generate the synthetic history; the forecast model has to rediscover these).
TRUE_REQUEST_BIAS = {"DRAM_A": 0.97, "DRAM_B": 0.84, "NAND_C": 0.95, "HBM_TV": 0.80, "SHORT": 1.12}
HISTORY_QUARTERS = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2",
                    "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]
PROGRAM_START = {"DRAM_A": "2024Q1", "DRAM_B": "2024Q3", "NAND_C": "2024Q1",
                 "HBM_TV": "2025Q3", "SHORT": "2024Q1"}

LOADING_CEILING = 0.85   # plan to keep tool groups below this loading
