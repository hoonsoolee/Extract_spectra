"""
Generate a formatted PDF user manual using matplotlib.PdfPages.
Run: python _make_manual_pdf.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import textwrap

OUTPUT = "Hyperspectral_Analysis_User_Manual.pdf"

# -- Colour palette ------------------------------------------------------------
C_DARK   = "#1a3a5c"   # header / title
C_GREEN  = "#2e7d62"   # accent
C_LIGHT  = "#f0f4f8"   # section bg
C_BODY   = "#222222"
C_MUTED  = "#666666"
C_WARN   = "#e67e22"
C_BOX    = "#eaf4ef"   # tip box bg

PAGE_W   = 8.27        # A4 inches
PAGE_H   = 11.69


def new_page(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
    return fig


def save(fig, pdf):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def header_bar(ax, title, subtitle=""):
    ax.set_facecolor(C_DARK)
    ax.text(0.04, 0.62, title,  color="white", fontsize=17, fontweight="bold",
            va="center", transform=ax.transAxes)
    if subtitle:
        ax.text(0.04, 0.25, subtitle, color="#b0c4d8", fontsize=9,
                va="center", transform=ax.transAxes)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off")


def section_title(ax, y, text, color=C_GREEN):
    ax.text(0.0, y, text, color=color, fontsize=12, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.plot([0, 1], [y - 0.025, y - 0.025],
            color=color, linewidth=1.2, transform=ax.transAxes)


def body_text(ax, y, text, fontsize=8.5, color=C_BODY, indent=0.0, wrap=110):
    lines = []
    for raw in text.split("\n"):
        lines.extend(textwrap.wrap(raw, wrap) if raw.strip() else [""])
    for line in lines:
        ax.text(indent, y, line, color=color, fontsize=fontsize,
                transform=ax.transAxes, va="top")
        y -= 0.033
    return y


def bullet(ax, y, items, fontsize=8.5, indent=0.03, spacing=0.033):
    for item in items:
        ax.text(indent, y, f"?  {item}", color=C_BODY, fontsize=fontsize,
                transform=ax.transAxes, va="top")
        y -= spacing
    return y


def tip_box(ax, y, text, height=0.07, color=C_BOX, label="TIP"):
    rect = FancyBboxPatch((0, y - height), 1, height,
                          boxstyle="round,pad=0.01",
                          linewidth=0.8, edgecolor=C_GREEN,
                          facecolor=color,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(0.015, y - 0.012, f">> {label}:", color=C_GREEN, fontsize=8,
            fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.015, y - 0.035, text, color=C_BODY, fontsize=8,
            transform=ax.transAxes, va="top",
            wrap=True)
    return y - height - 0.015


def page_number(fig, n):
    fig.text(0.5, 0.015, str(n), ha="center", fontsize=8, color=C_MUTED)


def table(ax, y, headers, rows, col_widths, fontsize=8):
    x0 = 0.0
    row_h = 0.032
    # header row
    cx = x0
    for h, w in zip(headers, col_widths):
        rect = FancyBboxPatch((cx, y - row_h), w - 0.005, row_h,
                              boxstyle="square,pad=0", lw=0,
                              facecolor=C_DARK, transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(cx + 0.008, y - row_h/2, h, color="white",
                fontsize=fontsize, fontweight="bold",
                va="center", transform=ax.transAxes)
        cx += w
    y -= row_h
    for ri, row in enumerate(rows):
        bg = "#f7faf9" if ri % 2 == 0 else "white"
        cx = x0
        for val, w in zip(row, col_widths):
            rect = FancyBboxPatch((cx, y - row_h), w - 0.005, row_h,
                                  boxstyle="square,pad=0", lw=0.4,
                                  edgecolor="#dde", facecolor=bg,
                                  transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            ax.text(cx + 0.008, y - row_h/2, val, color=C_BODY,
                    fontsize=fontsize - 0.5, va="center", transform=ax.transAxes)
            cx += w
        y -= row_h
    return y - 0.015


# ???????????????????????????????????????????????????????????????????????????????
# Build PDF
# ???????????????????????????????????????????????????????????????????????????????

with PdfPages(OUTPUT) as pdf:

    # ------------------------------- COVER -----------------------------------
    fig = new_page(pdf)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white"); ax.axis("off")

    # top colour block
    rect = FancyBboxPatch((0, 0.72), 1, 0.28,
                          boxstyle="square,pad=0", lw=0,
                          facecolor=C_DARK, transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(0.08, 0.93, "Hyperspectral Field Crop", color="white",
            fontsize=26, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.08, 0.85, "Analysis Tool", color="#a8d8c8",
            fontsize=22, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.08, 0.755, "Complete User Manual", color="#b0c4d8",
            fontsize=12, transform=ax.transAxes, va="top")

    ax.plot([0.08, 0.92], [0.70, 0.70], color=C_GREEN, linewidth=2, transform=ax.transAxes)

    ax.text(0.08, 0.67, "Version: Unsupervised Edition", color=C_MUTED,
            fontsize=9, transform=ax.transAxes, va="top")

    # description block
    desc = (
        "This manual provides step-by-step instructions for installing and operating the "
        "Hyperspectral Field Crop Analysis Tool. The tool automatically classifies pixels "
        "in hyperspectral images into vegetation, shadow, soil, and background classes, "
        "and extracts per-class reflectance spectra -- with no labelled training data required."
    )
    y = 0.60
    for line in textwrap.wrap(desc, 80):
        ax.text(0.08, y, line, color=C_BODY, fontsize=10,
                transform=ax.transAxes, va="top")
        y -= 0.04

    # feature badges
    features = [
        "[Hybrid] Hybrid (NDVI + Brightness + K-means)",
        "[K-Means] K-Means Clustering",
        "[SAM] Spectral Angle Mapping (SAM)",
        "[Autoencoder] Autoencoder (Deep Learning)",
        "[Report] Interactive HTML Report",
        "[Chart] Per-Class Spectra CSV Export",
    ]
    y = 0.44
    for feat in features:
        rect = FancyBboxPatch((0.08, y - 0.028), 0.60, 0.030,
                              boxstyle="round,pad=0.005", lw=0.6,
                              edgecolor=C_GREEN, facecolor=C_BOX,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.095, y - 0.006, feat, color=C_DARK, fontsize=9,
                transform=ax.transAxes, va="top")
        y -= 0.042

    ax.text(0.08, 0.13,
            "GitHub:  https://github.com/hoonsoolee/Extract_spectra",
            color=C_MUTED, fontsize=9, transform=ax.transAxes, va="top",
            style="italic")
    ax.text(0.08, 0.08,
            "Runs locally on CPU  ?  Windows / macOS / Linux  ?  Python 3.9+",
            color=C_MUTED, fontsize=9, transform=ax.transAxes, va="top")

    save(fig, pdf)

    # ---------------------------- TABLE OF CONTENTS ---------------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0]); header_bar(ax_h, "Table of Contents")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    toc = [
        ("1", "Installation",                         "3"),
        ("2", "Starting the Application",             "4"),
        ("3", "Sidebar Settings -- Data Source",       "5"),
        ("4", "Sidebar Settings -- Processing Mode",   "6"),
        ("5", "Classification Methods",               "7"),
        ("  5.1", "Hybrid (NDVI + Brightness + K-means)", "7"),
        ("  5.2", "K-Means",                          "9"),
        ("  5.3", "Spectral Angle Mapping (SAM)",     "10"),
        ("  5.4", "Autoencoder",                      "11"),
        ("6", "Number of Classes",                    "12"),
        ("7", "Output Settings",                      "12"),
        ("8", "Running the Analysis",                 "13"),
        ("9", "Understanding the HTML Report",        "14"),
        ("  9.1", "Image Overview",                   "14"),
        ("  9.2", "Per-Class Classification Images",  "15"),
        ("  9.3", "Classification Summary Table",     "15"),
        ("  9.4", "Reflectance Spectra Chart",        "16"),
        ("  9.5", "Quality Assessment",               "16"),
        ("  9.6", "Vegetation Separation Assessment", "17"),
        ("10", "Output Files",                        "18"),
        ("11", "Troubleshooting",                     "19"),
        ("12", "Quick Reference",                     "20"),
    ]

    y = 0.96
    for num, title, pg in toc:
        is_main = not num.startswith(" ")
        fw = "bold" if is_main else "normal"
        fs = 10 if is_main else 9
        col = C_DARK if is_main else C_BODY
        indent = 0.04 if is_main else 0.09
        ax.text(indent, y, f"{num}.", color=col, fontsize=fs,
                fontweight=fw, transform=ax.transAxes, va="top")
        ax.text(0.14, y, title, color=col, fontsize=fs,
                fontweight=fw, transform=ax.transAxes, va="top")
        ax.text(0.92, y, pg, color=C_MUTED, fontsize=fs,
                transform=ax.transAxes, va="top", ha="right")
        if is_main:
            ax.plot([0.04, 0.93], [y - 0.022, y - 0.022],
                    color="#e0e0e0", linewidth=0.5, transform=ax.transAxes)
        y -= 0.038 if is_main else 0.032

    page_number(fig, 2)
    save(fig, pdf)

    # ------------------------------ PAGE 3 -- INSTALLATION ---------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "1.  Installation", "Set up the tool on your computer -- one-time process")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    section_title(ax, y, "1.1  Requirements"); y -= 0.045
    rows_req = [
        ["Python 3.9 or later", "Programming language runtime  (download from python.org)"],
        ["Git",                 "Version control tool  (download from git-scm.com)"],
        ["8 GB RAM",            "Recommended minimum for typical field images"],
        ["PyTorch (optional)",  "Required only for the Autoencoder method"],
    ]
    y = table(ax, y, ["Requirement", "Notes"], rows_req,
              [0.32, 0.68], fontsize=8.5)
    y -= 0.02

    section_title(ax, y, "1.2  Installation Steps"); y -= 0.045

    steps = [
        ("Step 1 -- Clone the repository",
         'git clone https://github.com/hoonsoolee/Extract_spectra.git\ncd Extract_spectra'),
        ("Step 2 -- Install Python dependencies",
         'pip install -r requirements.txt'),
        ("Step 3 -- (Optional) Install PyTorch for Autoencoder method",
         'pip install torch'),
        ("Step 4 -- Verify installation",
         'python -m streamlit run app_unsupervised.py'),
    ]

    for title_s, code in steps:
        ax.text(0.0, y, title_s, color=C_DARK, fontsize=9, fontweight="bold",
                transform=ax.transAxes, va="top")
        y -= 0.032
        # code box
        n_lines = code.count("\n") + 1
        box_h = 0.028 * n_lines + 0.016
        rect = FancyBboxPatch((0.01, y - box_h), 0.98, box_h,
                              boxstyle="round,pad=0.005", lw=0.5,
                              edgecolor="#ccc", facecolor="#f5f5f5",
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        for i, cl in enumerate(code.split("\n")):
            ax.text(0.025, y - 0.01 - i * 0.028, cl,
                    color="#1a1a2e", fontsize=8.5, fontfamily="monospace",
                    transform=ax.transAxes, va="top")
        y -= box_h + 0.022

    y = tip_box(ax, y,
                "After running Step 4, your browser should open automatically at "
                "http://localhost:8501. If it does not, open that address manually.",
                height=0.065)

    section_title(ax, y, "1.3  Supported File Formats"); y -= 0.045
    rows_fmt = [
        ["ENVI",     ".hdr",          "Header file + matching data file (.raw / .bil / .bip / .bsq)"],
        ["GeoTIFF",  ".tif / .tiff",  "Standard geospatial raster format"],
        ["HDF5",     ".h5 / .hdf5",   "Hierarchical Data Format -- common in remote sensing"],
        ["MATLAB",   ".mat",          "MATLAB data file"],
    ]
    y = table(ax, y, ["Format", "Extension", "Notes"], rows_fmt,
              [0.15, 0.22, 0.63], fontsize=8.5)

    page_number(fig, 3)
    save(fig, pdf)

    # ---------------------- PAGE 4 -- STARTING THE APP -------------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "2.  Starting the Application")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    section_title(ax, y, "2.1  Launch Command"); y -= 0.045

    apps = [
        ("Unsupervised Edition  (recommended for most users)",
         "python -m streamlit run app_unsupervised.py",
         "Hybrid ? K-Means ? SAM ? Autoencoder  --  no labels required"),
        ("Full Edition  (includes supervised methods)",
         "python -m streamlit run app_en.py",
         "All methods including Random Forest and 1D-CNN  --  labels required for supervised"),
    ]
    for app_name, cmd, note in apps:
        ax.text(0.0, y, app_name, color=C_DARK, fontsize=9, fontweight="bold",
                transform=ax.transAxes, va="top"); y -= 0.032
        rect = FancyBboxPatch((0.01, y - 0.032), 0.98, 0.032,
                              boxstyle="round,pad=0.005", lw=0.5,
                              edgecolor="#ccc", facecolor="#f5f5f5",
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.025, y - 0.009, cmd, color="#1a1a2e", fontsize=9,
                fontfamily="monospace", transform=ax.transAxes, va="top")
        y -= 0.040
        ax.text(0.03, y, f"->  {note}", color=C_MUTED, fontsize=8,
                transform=ax.transAxes, va="top"); y -= 0.038

    y -= 0.01
    section_title(ax, y, "2.2  The Interface Layout"); y -= 0.045

    body = (
        "Once the app opens in your browser you will see two main areas:"
    )
    y = body_text(ax, y, body); y -= 0.01

    layout_items = [
        "LEFT SIDEBAR  --  All settings and parameters. You configure everything here before clicking Run.",
        "MAIN AREA  --  Displays the current configuration summary, run button results, classification map previews, and execution logs.",
    ]
    y = bullet(ax, y, layout_items, spacing=0.038); y -= 0.015

    # diagram
    diag_y = y - 0.22
    # sidebar box
    rect = FancyBboxPatch((0.01, diag_y), 0.28, 0.20,
                          boxstyle="round,pad=0.01", lw=1,
                          edgecolor=C_GREEN, facecolor=C_BOX,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(0.15, diag_y + 0.165, "SIDEBAR", color=C_GREEN,
            fontsize=9, fontweight="bold", ha="center",
            transform=ax.transAxes)
    for i, item in enumerate(["[Folder] Data Source", "[Mode] Processing Mode",
                               "[Method] Method", "[Classes] Classes",
                               "[Params] Parameters", "[Output] Output",
                               ">> Run Button"]):
        ax.text(0.04, diag_y + 0.13 - i * 0.022, item, color=C_BODY,
                fontsize=7.5, transform=ax.transAxes)

    # main area box
    rect2 = FancyBboxPatch((0.32, diag_y), 0.67, 0.20,
                           boxstyle="round,pad=0.01", lw=1,
                           edgecolor=C_DARK, facecolor=C_LIGHT,
                           transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect2)
    ax.text(0.655, diag_y + 0.165, "MAIN AREA", color=C_DARK,
            fontsize=9, fontweight="bold", ha="center",
            transform=ax.transAxes)
    for i, item in enumerate(["? Configuration summary card",
                               "Results after run:",
                               "  ? HTML report path",
                               "  ? Classification map preview",
                               "  ? Run log"]):
        ax.text(0.34, diag_y + 0.13 - i * 0.022, item, color=C_BODY,
                fontsize=7.5, transform=ax.transAxes)

    y = diag_y - 0.025

    y = tip_box(ax, y,
                "You do NOT need to restart the app between runs. Change any setting "
                "and click Run Analysis again -- results will update immediately.",
                height=0.065)

    section_title(ax, y, "2.3  Placing Your Data"); y -= 0.045
    body2 = (
        "Create a folder anywhere on your computer (e.g. C:/data/hyperspectral) "
        "and copy your hyperspectral image files into it. "
        "Subfolders are supported -- the tool will scan recursively."
        "\n\nFor ENVI format files, make sure the .hdr file and its matching data file "
        "(.raw, .bil, .bip, or .bsq) are in the same folder. "
        "The tool identifies them automatically by matching filenames."
    )
    y = body_text(ax, y, body2)

    page_number(fig, 4)
    save(fig, pdf)

    # ---------------------- PAGE 5 -- DATA SOURCE ------------------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "3.  Data Source", "Tell the tool where your image files are")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    body_text(ax, y,
        "The Data Source section is the first setting in the sidebar. "
        "Choose between two options:")
    y -= 0.06

    # Option boxes
    for title_o, color_o, items_o in [
        ("Option A -- Local Folder  (recommended)", C_GREEN, [
            "Select 'Local Folder' using the radio button.",
            "Type the full path to your data folder in the text box.",
            "  Example:  C:/data/hyperspectral   or   ./data",
            "The tool will scan this folder (and all subfolders) for supported files.",
        ]),
        ("Option B -- GitHub Repository", C_DARK, [
            "Select 'GitHub Repository' using the radio button.",
            "Repository: enter in the format  owner/repo  (e.g. mylab/crop_images)",
            "Sub-folder: enter the path inside the repo where images are stored.",
            "GitHub Token: required for private repositories only.",
            "  -> Generate at github.com/settings/tokens  (scope: repo)",
        ]),
    ]:
        rect = FancyBboxPatch((0.0, y - 0.155), 1, 0.155,
                              boxstyle="round,pad=0.01", lw=1,
                              edgecolor=color_o, facecolor=C_LIGHT,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.02, y - 0.015, title_o, color=color_o, fontsize=10,
                fontweight="bold", transform=ax.transAxes, va="top")
        yy = y - 0.05
        for item in items_o:
            indent = 0.045 if not item.startswith("  ") else 0.07
            prefix = "?  " if not item.startswith("  ") else ""
            ax.text(indent, yy, f"{prefix}{item.strip()}", color=C_BODY,
                    fontsize=8.5, transform=ax.transAxes, va="top")
            yy -= 0.027
        y -= 0.175

    y -= 0.01
    tip_box(ax, y,
            "If you are using the tool for the first time, choose 'Local Folder' "
            "and put a single test image in the ./data folder. "
            "This is the simplest and most reliable way to get started.",
            height=0.065)

    page_number(fig, 5)
    save(fig, pdf)

    # ---------------------- PAGE 6 -- PROCESSING MODE --------------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "4.  Processing Mode",
               "Choose whether to analyse one file or all files at once")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    body_text(ax, y,
        "The Processing Mode section appears directly below Data Source in the sidebar. "
        "It controls how many files are processed in a single run.")
    y -= 0.07

    for mode, color_m, steps_m, when_m in [
        ("[Single]  Single File Mode", C_GREEN,
         [
             "1.  Select 'Single File' from the radio buttons.",
             "2.  Click the 'Scan Folder' button.",
             "    -> The tool scans your data folder and lists all supported files.",
             "3.  Use the dropdown menu to select the file you want to analyse.",
             "4.  Click 'Run Analysis' in the sidebar.",
             "5.  Results are saved to:  output/<filename>/",
         ],
         "Use this mode when you want to inspect one specific image in detail, "
         "or when testing settings before processing a large batch."),
        ("[Batch]  Batch Mode (all files)", C_DARK,
         [
             "1.  Select 'Batch (all files)' from the radio buttons.",
             "2.  Click 'Run Analysis' in the sidebar.",
             "    -> All supported files in your data folder are processed in order.",
             "3.  Each file gets its own output subfolder and its own HTML report.",
             "4.  File processing order is alphabetical by filename.",
         ],
         "Use this mode when you want to process an entire dataset. "
         "Each file produces an independent report, making it easy to "
         "compare results across images."),
    ]:
        rect = FancyBboxPatch((0.0, y - 0.215), 1, 0.215,
                              boxstyle="round,pad=0.01", lw=1,
                              edgecolor=color_m, facecolor=C_LIGHT,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.02, y - 0.015, mode, color=color_m, fontsize=10.5,
                fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(0.02, y - 0.045, f"When to use:  {when_m}", color=C_MUTED,
                fontsize=8, transform=ax.transAxes, va="top",
                style="italic", wrap=True)
        yy = y - 0.085
        for s in steps_m:
            indent = 0.03 if not s.startswith("    ") else 0.055
            ax.text(indent, yy, s.strip(), color=C_BODY, fontsize=8.5,
                    transform=ax.transAxes, va="top")
            yy -= 0.026
        y -= 0.235

    y -= 0.01
    body_text(ax, y,
        "Processing time is displayed in the success banner after each run. "
        "Use the Single File time as a rough estimate of how long the full batch will take: "
        "total time ? single-file time x number of files.",
        color=C_MUTED)

    page_number(fig, 6)
    save(fig, pdf)

    # ---------------------- PAGE 7-8 -- HYBRID METHOD -------------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "5.  Classification Methods",
               "How the algorithm groups pixels into classes")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    body_text(ax, y,
        "The Classification Method section is where you choose the core algorithm. "
        "All four methods available in this edition are fully unsupervised -- "
        "they discover natural groupings in the data automatically, "
        "with no manually labelled training samples required.")
    y -= 0.07

    section_title(ax, y, "5.1  Hybrid  (Recommended Default)"); y -= 0.045

    body_text(ax, y,
        "The Hybrid method combines three complementary techniques in sequence. "
        "It is the recommended starting point for most field crop images because "
        "it leverages well-known vegetation physics rather than relying purely on "
        "statistical grouping.")
    y -= 0.085

    # pipeline diagram
    stages = [
        ("Step 1\nNDVI Mask", C_GREEN,
         "Computes NDVI = (NIR ? Red) / (NIR + Red)\nfor every pixel.\n"
         "Pixels above the NDVI threshold -> Vegetation.\nAll others -> Non-vegetation."),
        ("Step 2\nBrightness\nMask", C_DARK,
         "Within the vegetation pixels,\ncomputes mean reflectance across all bands.\n"
         "Low brightness -> Shadow.\nHigh brightness -> Sunlit leaves."),
        ("Step 3\nK-Means\nRefinement", "#8e44ad",
         "Within each segment (Sunlit / Shadow / Soil),\nruns K-means clustering "
         "to find sub-classes.\nThis separates e.g. different crop types or\n"
         "soil moisture levels within each segment."),
    ]
    sx = 0.01
    box_w = 0.30
    box_h = 0.20
    for stage_title, stage_color, stage_desc in stages:
        rect = FancyBboxPatch((sx, y - box_h), box_w, box_h,
                              boxstyle="round,pad=0.01", lw=1.2,
                              edgecolor=stage_color, facecolor=C_LIGHT,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(sx + box_w/2, y - 0.02, stage_title, color=stage_color,
                fontsize=9, fontweight="bold", ha="center",
                transform=ax.transAxes, va="top")
        desc_y = y - 0.065
        for dline in stage_desc.split("\n"):
            ax.text(sx + 0.015, desc_y, dline, color=C_BODY, fontsize=7.5,
                    transform=ax.transAxes, va="top")
            desc_y -= 0.025
        if sx + box_w < 0.95:
            ax.annotate("", xy=(sx + box_w + 0.03, y - box_h/2),
                        xytext=(sx + box_w, y - box_h/2),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.5))
        sx += box_w + 0.045

    y -= box_h + 0.025

    section_title(ax, y, "Hybrid Parameters"); y -= 0.04

    rows_h = [
        ["NDVI Threshold", "0.15",
         "Pixels with NDVI ? this value are classified as vegetation (leaves). "
         "Increase if non-vegetation pixels are being included; "
         "decrease if too few leaf pixels are detected."],
        ["Brightness Threshold", "0.08",
         "Within vegetation pixels, those with mean reflectance below this value "
         "are classified as shadow. Increase in strongly shaded images; "
         "decrease in bright/overcast conditions."],
    ]
    y = table(ax, y, ["Parameter", "Default", "What it does"], rows_h,
              [0.22, 0.10, 0.68], fontsize=8)
    y -= 0.01

    # class id table
    section_title(ax, y, "Output Class IDs (Hybrid)"); y -= 0.04
    rows_ids = [
        ["0", "Background",      "Non-vegetation, non-soil areas (e.g. path edges, sky)"],
        ["1", "Sunlit Leaves",   "Vegetation pixels with high reflectance (direct sunlight)"],
        ["2", "Shadowed Leaves", "Vegetation pixels with low reflectance (in shadow)"],
        ["3", "Soil",            "Bare soil and ground cover"],
        ["4", "Other",           "Remaining unclassified pixels"],
    ]
    y = table(ax, y, ["ID", "Class Name", "Description"], rows_ids,
              [0.07, 0.20, 0.73], fontsize=8)

    page_number(fig, 7)
    save(fig, pdf)

    # ---------------------- PAGE 8 -- WHEN TO ADJUST HYBRID -------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "5.1  Hybrid -- Tuning Guide")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    section_title(ax, y, "When to Adjust the NDVI Threshold"); y -= 0.04

    rows_ndvi = [
        ["Too many non-plant\npixels in leaf classes",
         "Increase NDVI threshold\n(e.g. 0.15 -> 0.25)",
         "The current threshold is too permissive -- bare soil or path "
         "pixels have NDVI slightly above zero."],
        ["Too few leaf pixels\ndetected",
         "Decrease NDVI threshold\n(e.g. 0.15 -> 0.08)",
         "Stressed or senescent crops may have lower NDVI. "
         "Lowering the threshold captures more marginal vegetation."],
        ["Image has no NIR band",
         "Use K-Means or SAM\ninstead",
         "NDVI requires a Near-Infrared (~800 nm) and Red (~660 nm) band. "
         "If those bands are absent, the algorithm automatically falls back to K-Means."],
    ]
    y = table(ax, y, ["Symptom", "Action", "Explanation"], rows_ndvi,
              [0.26, 0.25, 0.49], fontsize=8)
    y -= 0.025

    section_title(ax, y, "When to Adjust the Brightness Threshold"); y -= 0.04
    rows_br = [
        ["Shadow class contains\nmany sunlit pixels",
         "Increase brightness\nthreshold",
         "The tool is classifying too many pixels as shadow. "
         "Raise the threshold until only genuinely dark pixels are included."],
        ["Sunlit class contains\nnoticeable shadows",
         "Decrease brightness\nthreshold",
         "The tool is not separating shadows well. "
         "Lower the threshold to pull more dark pixels into the shadow class."],
    ]
    y = table(ax, y, ["Symptom", "Action", "Explanation"], rows_br,
              [0.26, 0.25, 0.49], fontsize=8)
    y -= 0.015

    tip_box(ax, y,
            "A quick workflow: run once with default settings, open the HTML report, "
            "and look at the Per-Class Images section. If a class clearly contains "
            "mixed pixel types, adjust the relevant threshold and re-run.",
            height=0.07)
    y -= 0.10

    section_title(ax, y, "5.2  K-Means"); y -= 0.04
    body_text(ax, y,
        "K-Means is a classic unsupervised clustering algorithm. "
        "It requires no prior knowledge about the image content and makes no assumptions "
        "about vegetation indices.")
    y -= 0.075

    body_text(ax, y, "How it works:", color=C_DARK, fontsize=9)
    y -= 0.035
    kmeans_steps = [
        "PCA (Principal Component Analysis) reduces the hundreds of spectral bands "
        "to 15 principal components, capturing most of the variance while discarding noise.",
        "K-Means partitions the compressed spectral space into K clusters by "
        "iteratively minimising the distance from each pixel to its cluster centroid.",
        "Each cluster corresponds to one output class. "
        "The algorithm assigns meaningful colours but not names -- "
        "you interpret the classes from the spectral chart in the report.",
    ]
    y = bullet(ax, y, kmeans_steps, spacing=0.04)
    y -= 0.02

    rows_km = [
        ["Number of Classes", "6",
         "How many clusters to find. More clusters = finer separation but "
         "harder to interpret. Start with 6 and adjust based on the report."],
    ]
    y = table(ax, y, ["Parameter", "Default", "What it does"], rows_km,
              [0.22, 0.10, 0.68], fontsize=8)
    y -= 0.02

    tip_box(ax, y,
            "K-Means is best when you want a purely data-driven result with no "
            "vegetation-index assumptions. It works even on images without NIR bands. "
            "If you are not sure which method to use and do not have NIR coverage, "
            "start with K-Means.",
            height=0.08)

    page_number(fig, 8)
    save(fig, pdf)

    # ---------------------- PAGE 9 -- SAM --------------------------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "5.3  Spectral Angle Mapping (SAM)",
               "Illumination-invariant classification based on spectral shape")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    body_text(ax, y,
        "SAM measures the similarity between two spectra by computing the angle "
        "between their vectors in the spectral feature space, rather than their "
        "Euclidean distance. This makes SAM insensitive to illumination intensity -- "
        "a sunlit and a shadowed pixel of the same material will have the same "
        "spectral angle, even though their absolute reflectance values differ significantly.")
    y -= 0.1

    section_title(ax, y, "How SAM Works"); y -= 0.04
    sam_steps = [
        "Endmembers (representative spectra) are automatically extracted from the "
        "image using PCA -- no user-provided reference spectra needed.",
        "Each pixel's spectrum is compared to every endmember by computing the "
        "spectral angle (in radians). A small angle means high similarity.",
        "Each pixel is assigned to the class of the endmember with the smallest angle.",
        "If the smallest angle exceeds the angle threshold, the pixel is assigned "
        "to Background (class 0) -- it does not resemble any known class.",
    ]
    y = bullet(ax, y, sam_steps, spacing=0.04)
    y -= 0.02

    rows_sam = [
        ["Angle Threshold", "0.10 rad\n(? 5.7deg)",
         "Maximum allowed angle between a pixel and its nearest endmember. "
         "Pixels with angle > threshold -> Background. "
         "Lower values = stricter matching; raise if too many pixels are Background."],
        ["Number of Classes", "6",
         "Number of endmembers to extract. Each endmember becomes one class."],
    ]
    y = table(ax, y, ["Parameter", "Default", "What it does"], rows_sam,
              [0.22, 0.16, 0.62], fontsize=8)
    y -= 0.015

    tip_box(ax, y,
            "SAM is particularly useful for images with strong shadow gradients "
            "or varying illumination across the scene (e.g. images taken near sunrise "
            "or sunset, or in partially cloudy conditions).",
            height=0.07)
    y -= 0.10

    section_title(ax, y, "5.4  Autoencoder  (Deep Learning)"); y -= 0.04
    body_text(ax, y,
        "The Autoencoder method uses a small neural network to learn a compressed "
        "representation of each pixel's spectrum. K-Means clustering is then applied "
        "in this learned latent space, which can capture non-linear spectral relationships "
        "that PCA (used by K-Means method) might miss.")
    y -= 0.09

    section_title(ax, y, "How the Autoencoder Works"); y -= 0.04
    ae_steps = [
        "An MLP Encoder compresses each spectrum (hundreds of bands) "
        "down to 16 latent dimensions.",
        "An MLP Decoder reconstructs the original spectrum from the 16 dimensions. "
        "The network is trained to minimise reconstruction error.",
        "After training, the encoder output (16-dim latent vector) is used as the "
        "feature for each pixel.",
        "K-Means clustering is applied to the latent vectors to produce final classes.",
    ]
    y = bullet(ax, y, ae_steps, spacing=0.04)
    y -= 0.02

    rows_ae = [
        ["Training Epochs", "60",
         "Number of passes over the training data. More epochs = better compression "
         "but longer runtime. 60 is sufficient for most images; "
         "reduce to 20-30 for a quick test."],
        ["Number of Classes", "6",
         "Number of K-Means clusters applied to the latent space."],
    ]
    y = table(ax, y, ["Parameter", "Default", "What it does"], rows_ae,
              [0.22, 0.10, 0.68], fontsize=8)
    y -= 0.015

    ax.text(0.0, y, "WARNING:  Requirement:", color=C_WARN, fontsize=9,
            fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.0, y - 0.032,
            "PyTorch must be installed:   pip install torch",
            color=C_BODY, fontsize=8.5, fontfamily="monospace",
            transform=ax.transAxes, va="top")

    page_number(fig, 9)
    save(fig, pdf)

    # ---------------------- PAGE 10 -- CLASSES / OUTPUT / RUNNING -------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "6 - 8.  Classes, Output & Running the Analysis")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    section_title(ax, y, "6.  Number of Classes"); y -= 0.04
    body_text(ax, y,
        "The Number of Classes slider controls how many groups the algorithm will create. "
        "The optimal value depends on the complexity of your scene:")
    y -= 0.06
    rows_nc = [
        ["2 - 3", "Very simple scenes. One crop type, minimal background."],
        ["4 - 6", "Typical field images. Recommended starting range."],
        ["7 - 10","Complex scenes: multiple crop types, varied soil, deep shadow patches."],
        ["11+",   "Use only if you have a specific reason. Hard to interpret."],
    ]
    y = table(ax, y, ["Value", "When to use"], rows_nc,
              [0.15, 0.85], fontsize=8.5)
    y -= 0.015
    tip_box(ax, y,
            "Start with 6 classes. Open the HTML report and check the Per-Class Images. "
            "If two classes look visually identical, reduce the count. "
            "If one class clearly contains mixed content, increase it.",
            height=0.07)
    y -= 0.10

    section_title(ax, y, "7.  Output Settings"); y -= 0.04
    rows_out = [
        ["Output Folder", "./output",
         "Path where all results are saved. "
         "A subfolder named after each image file is created automatically."],
        ["File Limit", "0 (= all)",
         "Limits the number of files processed. "
         "Set to 1 or 2 during initial testing to save time."],
        ["Verbose Logging", "Off",
         "Shows DEBUG-level messages in the run log. "
         "Useful for diagnosing errors; leave off for normal use."],
    ]
    y = table(ax, y, ["Setting", "Default", "What it does"], rows_out,
              [0.20, 0.12, 0.68], fontsize=8)
    y -= 0.02

    section_title(ax, y, "8.  Running the Analysis"); y -= 0.04
    run_steps = [
        "Configure all settings in the sidebar (Data Source, Mode, Method, Parameters, Output).",
        "Click the blue '>> Run Analysis' button at the bottom of the sidebar.",
        "A spinner appears: 'Analysing? (may take several minutes for large images)'.",
        "When finished, a green banner shows:  OK Analysis complete!  ?  Time: Xs",
        "The HTML report path is displayed -- open it in any web browser.",
        "A preview of the classification map is shown directly in the app.",
        "The run log at the bottom shows per-step timing and quality metrics.",
    ]
    y = bullet(ax, y, run_steps, spacing=0.038)
    y -= 0.015

    # timing guide
    section_title(ax, y, "Typical Processing Times (CPU, 500 x 500 px, 200 bands)"); y -= 0.04
    rows_time = [
        ["Hybrid",      "30 - 90 s",  "Depends on number of sub-clusters"],
        ["K-Means",     "20 - 60 s",  "Faster; no vegetation masking"],
        ["SAM",         "30 - 90 s",  "Similar to Hybrid"],
        ["Autoencoder", "3 - 10 min", "Dominated by neural network training"],
    ]
    y = table(ax, y, ["Method", "Approx. Time", "Notes"], rows_time,
              [0.20, 0.20, 0.60], fontsize=8)

    page_number(fig, 10)
    save(fig, pdf)

    # ---------------------- PAGE 11 -- HTML REPORT (1) -------------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "9.  Understanding the HTML Report  (Part 1)",
               "All results are saved in a single self-contained HTML file")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    body_text(ax, y,
        "After each run, a file named report_YYYYMMDD_HHMMSS.html is created in the output folder. "
        "Open this file in any web browser (Chrome, Firefox, Edge, Safari). "
        "It contains everything you need to evaluate the classification result.")
    y -= 0.075

    section_title(ax, y, "9.1  Image Overview"); y -= 0.04
    body_text(ax, y,
        "The first card shows three side-by-side images of the scene:"); y -= 0.04
    overview_items = [
        "RGB Composite  --  A natural-colour view of the field, constructed from the "
        "Red (~660 nm), Green (~550 nm), and Blue (~450 nm) bands. "
        "This is what the scene looks like to the human eye.",

        "CIR False-Color Composite  --  Near-Infrared (NIR, ~800 nm) is mapped to red, "
        "Red to green, and Green to blue. "
        "Healthy green vegetation appears bright red/magenta in CIR composites. "
        "This is a standard remote sensing visualization for vegetation assessment.",

        "Classification Map  --  All classes overlaid on the image with their assigned colours. "
        "Each class has a colour-coded legend showing pixel count and percentage.",
    ]
    y = bullet(ax, y, overview_items, spacing=0.05)
    y -= 0.02

    tip_box(ax, y,
            "The CIR composite is the quickest way to verify that the Hybrid method's "
            "NDVI mask is working correctly. Bright red/magenta areas should align with "
            "the 'Sunlit Leaves' and 'Shadowed Leaves' classes in the classification map.",
            height=0.07)
    y -= 0.10

    section_title(ax, y, "9.2  Per-Class Classification Images"); y -= 0.04
    body_text(ax, y,
        "This section shows one large image per class. "
        "In each image, the pixels belonging to that class are shown in the class colour, "
        "while all other pixels are shown as a darkened greyscale background. "
        "This makes it easy to judge whether each class represents a meaningful, "
        "coherent group of pixels, or whether it has been incorrectly mixed with other materials.")
    y -= 0.095
    per_class_items = [
        "A well-classified image:  one class covers a spatially coherent region "
        "(e.g. the leaf canopy, or a patch of soil).",
        "A poorly classified image:  the class pixels are scattered randomly across the scene, "
        "or include obvious mixtures of vegetation and soil.",
    ]
    y = bullet(ax, y, per_class_items, spacing=0.04)
    y -= 0.015

    section_title(ax, y, "9.3  Classification Summary Table"); y -= 0.04
    body_text(ax, y,
        "A table listing each class with its pixel count and percentage of the total image area. "
        "A horizontal colour bar shows the proportion visually. "
        "Use this to quickly check whether the class distribution looks reasonable "
        "(e.g. soil should not be 90% of a dense canopy image).")
    y -= 0.085

    section_title(ax, y, "9.4  Reflectance Spectra Chart"); y -= 0.04
    body_text(ax, y,
        "An interactive Plotly chart showing the mean reflectance spectrum for each class "
        "plotted against wavelength (nm) or band index. "
        "The shaded band around each line represents +-1 standard deviation, "
        "indicating the spectral variability within the class.")
    y -= 0.09
    spectra_items = [
        "Hover over the chart to see exact reflectance values at each wavelength.",
        "Click a class name in the legend to show/hide that class.",
        "Double-click a class name to isolate it (hide all others).",
        "Use the toolbar in the top-right corner to zoom, pan, or download the chart as PNG.",
        "Well-separated spectra (clear vertical distance between lines) indicate "
        "that the classes are spectrally distinct -- a sign of good classification.",
    ]
    y = bullet(ax, y, spectra_items, spacing=0.035)

    page_number(fig, 11)
    save(fig, pdf)

    # ---------------------- PAGE 12 -- HTML REPORT (2) -------------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "9.  Understanding the HTML Report  (Part 2)",
               "Quality metrics and vegetation separation assessment")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    section_title(ax, y, "9.5  Quality Assessment"); y -= 0.04
    body_text(ax, y,
        "This section provides quantitative metrics to evaluate how well the "
        "clustering has separated the spectral classes. Two metrics are shown:")
    y -= 0.065

    for metric, default_interp, description, good, bad in [
        ("Silhouette Score  ( ?1 to +1 )",
         "Converted to a 0-100% Quality Gauge",
         "Measures how similar each pixel's spectrum is to its own class compared "
         "to the nearest other class. A value near +1 (100%) means pixels are "
         "well-matched to their class and far from others. "
         "A value near 0 (50%) means classes overlap. "
         "A negative value means pixels are closer to a different class.",
         "Above 75%  (Silhouette > +0.50)",
         "Below 50%  (Silhouette < 0)"),
        ("Davies-Bouldin Index  ( lower = better )",
         "Shown as a numeric value (no ideal range)",
         "Measures the average ratio of within-cluster scatter to between-cluster "
         "separation. A lower value indicates more compact and well-separated clusters.",
         "Below 1.0",
         "Above 2.0"),
    ]:
        rect = FancyBboxPatch((0.0, y - 0.135), 1, 0.135,
                              boxstyle="round,pad=0.01", lw=0.7,
                              edgecolor="#ccc", facecolor=C_LIGHT,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.02, y - 0.012, metric, color=C_DARK, fontsize=9.5,
                fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(0.02, y - 0.038, default_interp, color=C_GREEN, fontsize=8,
                style="italic", transform=ax.transAxes, va="top")
        desc_y = y - 0.062
        for dline in textwrap.wrap(description, 105):
            ax.text(0.02, desc_y, dline, color=C_BODY, fontsize=8,
                    transform=ax.transAxes, va="top")
            desc_y -= 0.025
        ax.text(0.60, y - 0.060, f"Good:  {good}", color="#27ae60",
                fontsize=8, transform=ax.transAxes, va="top")
        ax.text(0.60, y - 0.090, f"Poor:  {bad}", color="#e74c3c",
                fontsize=8, transform=ax.transAxes, va="top")
        y -= 0.150

    y -= 0.01
    section_title(ax, y, "9.6  Vegetation Separation Assessment"); y -= 0.04
    body_text(ax, y,
        "This section evaluates how accurately the algorithm has identified vegetation "
        "pixels, using NDVI as an independent ground truth. "
        "It answers the question: of all the pixels that NDVI says are vegetation, "
        "how many did the classifier correctly label as leaves?")
    y -= 0.09

    rows_veg = [
        ["Recall", "True positive rate",
         "Of all pixels with NDVI > 0.15, what fraction ended up in a leaf class? "
         "Low recall means leaf pixels are being missed and placed in soil/background."],
        ["Precision", "Positive predictive value",
         "Of all pixels placed in a leaf class, what fraction truly have NDVI > 0.15? "
         "Low precision means non-vegetation is being incorrectly labelled as leaves."],
        ["F1 Score", "Harmonic mean of\nRecall & Precision",
         "Single combined score. Above 0.85 is excellent. Below 0.60 suggests "
         "significant misclassification of vegetation vs. non-vegetation."],
    ]
    y = table(ax, y, ["Metric", "Definition", "How to interpret"], rows_veg,
              [0.14, 0.22, 0.64], fontsize=8)
    y -= 0.015

    tip_box(ax, y,
            "If F1 Score is low, check the Per-Class Images first. "
            "Then adjust the NDVI threshold (raise if non-vegetation is mixed into leaves; "
            "lower if genuine vegetation is being missed) and re-run.",
            height=0.07)

    page_number(fig, 12)
    save(fig, pdf)

    # ---------------------- PAGE 13 -- OUTPUT FILES / TROUBLESHOOTING ----------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "10 - 11.  Output Files & Troubleshooting")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    section_title(ax, y, "10.  Output Files"); y -= 0.04
    body_text(ax, y,
        "All output is saved inside the folder you specified in the Output Folder setting. "
        "A subfolder is created for each processed image file:")
    y -= 0.06

    # folder tree
    tree_lines = [
        ("output/",                              0.0,  C_DARK,  10, "bold"),
        ("+-- image_filename/",                 0.04, C_DARK,   9, "normal"),
        ("    +-- spectra.csv",                 0.08, C_GREEN,  9, "normal"),
        ("    +-- class_map.png",               0.08, C_GREEN,  9, "normal"),
        ("    +-- report_20240407_143022.html", 0.08, C_GREEN,  9, "normal"),
    ]
    rect = FancyBboxPatch((0.0, y - 0.17), 0.65, 0.17,
                          boxstyle="round,pad=0.01", lw=0.5,
                          edgecolor="#ccc", facecolor="#f5f5f5",
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ty = y - 0.02
    for text_t, ind, col, fs, fw in tree_lines:
        ax.text(0.02 + ind, ty, text_t, color=col, fontsize=fs,
                fontfamily="monospace", fontweight=fw,
                transform=ax.transAxes, va="top")
        ty -= 0.030
    y -= 0.185

    rows_files = [
        ["spectra.csv",
         "CSV table: one row per spectral band, columns = class mean reflectance and std. "
         "Load in Python (pandas.read_csv), R (read.csv), or Excel for further analysis."],
        ["class_map.png",
         "PNG image of the classification map with colour legend. "
         "Suitable for inclusion in reports or presentations."],
        ["report_*.html",
         "Self-contained HTML file with all visualizations and metrics. "
         "Open in any browser. No internet connection required."],
    ]
    y = table(ax, y, ["File", "Contents & use"], rows_files,
              [0.25, 0.75], fontsize=8)
    y -= 0.025

    section_title(ax, y, "11.  Troubleshooting"); y -= 0.04
    rows_ts = [
        ["ModuleNotFoundError",
         "Run:  pip install -r requirements.txt\n"
         "For Autoencoder:  pip install torch"],
        ["'Unsupported format' error",
         "Check the file extension is .hdr / .tif / .tiff / .h5 / .hdf5 / .mat"],
        ["ENVI file does not load",
         "Make sure the .hdr file and its data file (.raw/.bil/.bip/.bsq) "
         "are in the same folder with the same base name."],
        ["Classification map is mostly one colour",
         "Lower the NDVI threshold (Hybrid) or increase the number of classes."],
        ["Too many pixels in Background class (SAM)",
         "Increase the Angle Threshold (e.g. 0.10 -> 0.20)."],
        ["Processing is very slow",
         "Set spatial_downsample: 2 in config.yaml to halve the image dimensions, "
         "or reduce Training Epochs (Autoencoder)."],
        ["Browser does not open automatically",
         "Open http://localhost:8501 manually in your browser."],
        ["Port 8501 already in use",
         "Run:  python -m streamlit run app_unsupervised.py --server.port 8502"],
    ]
    for i, (prob, sol) in enumerate(rows_ts):
        bg = "#f7faf9" if i % 2 == 0 else "white"
        row_h_val = 0.055
        rect = FancyBboxPatch((0.0, y - row_h_val), 1, row_h_val,
                              boxstyle="square,pad=0", lw=0.4,
                              edgecolor="#dde", facecolor=bg,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.01, y - 0.01, prob, color=C_DARK, fontsize=8,
                fontweight="bold", transform=ax.transAxes, va="top")
        sy = y - 0.028
        for sline in textwrap.wrap(sol, 95):
            ax.text(0.01, sy, sline, color=C_BODY, fontsize=7.8,
                    transform=ax.transAxes, va="top")
            sy -= 0.020
        y -= row_h_val

    page_number(fig, 13)
    save(fig, pdf)

    # ---------------------- PAGE 14 -- QUICK REFERENCE -------------------------
    fig = new_page(pdf)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    ax_h = fig.add_subplot(gs[0])
    header_bar(ax_h, "12.  Quick Reference Card")
    ax = fig.add_subplot(gs[1]); ax.axis("off")

    y = 0.96
    section_title(ax, y, "Method Selection Guide"); y -= 0.04
    rows_sel = [
        ["I have no idea where to start",
         "Hybrid",  "6",   "Default settings"],
        ["My image has no NIR band",
         "K-Means", "6",   "Default settings"],
        ["Strong shadows / uneven lighting",
         "SAM",     "6",   "Angle threshold: 0.15"],
        ["Want deep learning features",
         "Autoencoder", "6", "Epochs: 60 (requires PyTorch)"],
        ["Need to compare methods",
         "Run all four", "--", "Use Single File mode for each"],
    ]
    y = table(ax, y, ["Situation", "Method", "Classes", "Parameter notes"],
              rows_sel, [0.35, 0.15, 0.12, 0.38], fontsize=8)
    y -= 0.03

    section_title(ax, y, "Parameter Quick Reference"); y -= 0.04
    rows_par = [
        ["NDVI Threshold",       "Hybrid",      "0.15",   "0.05 - 0.30",  "Vegetation sensitivity"],
        ["Brightness Threshold", "Hybrid",      "0.08",   "0.03 - 0.20",  "Shadow sensitivity"],
        ["Angle Threshold",      "SAM",         "0.10",   "0.05 - 0.30",  "Matching strictness"],
        ["Training Epochs",      "Autoencoder", "60",     "20 - 200",     "Neural network training"],
        ["Number of Classes",    "All methods", "6",      "2 - 20",       "Output cluster count"],
        ["File Limit",           "All modes",   "0 (all)","1 - N",        "For quick testing"],
    ]
    y = table(ax, y, ["Parameter", "Method", "Default", "Typical Range", "Controls"],
              rows_par, [0.25, 0.16, 0.11, 0.18, 0.30], fontsize=8)
    y -= 0.03

    section_title(ax, y, "Keyboard Shortcuts / Quick Actions in the App"); y -= 0.04
    rows_kb = [
        ["Change any setting -> click Run again", "Results update immediately; no restart needed"],
        ["File Limit = 1",                       "Process only the first file for a quick test"],
        ["Open .html in browser",                "All charts are interactive -- hover, zoom, toggle"],
        ["spectra.csv in Excel",                 "Open directly; rows = bands, columns = classes"],
    ]
    y = table(ax, y, ["Action", "Effect"], rows_kb, [0.45, 0.55], fontsize=8)
    y -= 0.03

    section_title(ax, y, "Quality Score Interpretation"); y -= 0.04
    rows_q = [
        ["90 - 100%", "Excellent", "Classes are very well separated"],
        ["75 - 90%",  "Good",      "Reliable classification for most purposes"],
        ["50 - 75%",  "Fair",      "Some class overlap; consider adjusting parameters"],
        ["Below 50%", "Poor",      "Significant overlap; try different method or class count"],
    ]
    y = table(ax, y, ["Quality Gauge", "Rating", "Meaning"],
              rows_q, [0.20, 0.15, 0.65], fontsize=8)

    # footer
    ax.plot([0.0, 1.0], [0.04, 0.04], color="#ddd", linewidth=0.8,
            transform=ax.transAxes)
    ax.text(0.5, 0.025,
            "GitHub: https://github.com/hoonsoolee/Extract_spectra",
            color=C_MUTED, fontsize=8, ha="center", style="italic",
            transform=ax.transAxes)

    page_number(fig, 14)
    save(fig, pdf)

print(f"Done -> {OUTPUT}")
