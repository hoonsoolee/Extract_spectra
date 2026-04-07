# Hyperspectral Crop Analysis Tool — Presentation Script

> Use this as a talking guide when demonstrating the tool to colleagues.
> Estimated time: 10–15 minutes.

---

## 1. Opening (30 seconds)

*"I'd like to show you a tool we built for analyzing hyperspectral field images of crops.
The goal is simple: given a raw hyperspectral image, automatically separate pixels into
meaningful classes — sunlit leaves, shadowed leaves, soil, and background —
and then extract the average reflectance spectrum for each class.
Everything runs locally on your machine, no cloud required, and no labels are needed."*

---

## 2. What Is a Hyperspectral Image? (1–2 minutes)

*"Before I show the tool, a quick refresher on the data we're working with."*

- A regular camera captures **3 channels** (R, G, B).
- A hyperspectral camera captures **hundreds of narrow spectral bands**, typically from ~400 nm (visible blue) up to ~2500 nm (shortwave infrared).
- Each pixel in the image holds a full **reflectance spectrum** — like a spectral fingerprint.
- This makes it possible to distinguish materials that look identical in a regular photo, such as healthy vs. water-stressed leaves, or different soil types.

*"Our images are from field crops. The sensor is mounted on a drone or a ground vehicle,
and we process images with spatial dimensions anywhere from a few hundred to several thousand pixels per side,
with 150 to 300 spectral bands."*

---

## 3. What the Tool Does — The Pipeline (2 minutes)

*"The processing chain has five steps:"*

```
Raw image  →  Preprocess  →  Classify  →  Extract spectra  →  Report
```

1. **Load** — Reads ENVI, GeoTIFF, HDF5, or MATLAB files.
2. **Preprocess** — Normalizes reflectance, removes bad bands (water absorption regions: 1340–1460 nm and 1790–1960 nm), and optionally applies Savitzky-Golay smoothing.
3. **Classify** — Groups pixels into classes using one of the unsupervised algorithms (described below).
4. **Extract spectra** — Computes the mean and standard deviation reflectance spectrum for each class.
5. **Report** — Generates a self-contained HTML file with all results, charts, and quality metrics.

---

## 4. Live Demo — Running the App (5–7 minutes)

### Starting the app

```bash
cd Extract_spectra
python -m streamlit run app_unsupervised.py
```

*"This opens a web interface in your browser. Everything you see here runs locally —
Streamlit is just a Python library that renders the controls as a webpage."*

---

### Step 1: Data Source

*"On the left sidebar, the first thing you set is where your data lives.
I'll use a local folder. Just type in the path to the folder containing your hyperspectral files."*

> Point to the **Data Source** section and type in the path.

---

### Step 2: Processing Mode

*"You have two options here:"*

- **Single File** — Click 'Scan Folder', select one file from the dropdown, and analyse just that one. This is great for quickly checking a result before committing to a full batch run.
- **Batch (all files)** — Processes every file in the folder sequentially. Each file gets its own output subfolder and its own HTML report.

*"For the demo, I'll use Single File so we get a result fast."*

> Click **Scan Folder**, select one file from the dropdown.

---

### Step 3: Classification Method

*"This is the core of the tool. We have four unsupervised methods — no labelled training data needed for any of them."*

**Hybrid (recommended default)**
> *"This is the method I recommend for most field crop images.
> It works in three stages:
> First, it computes NDVI to find vegetation pixels.
> Then it uses mean reflectance to separate sunlit from shadowed areas.
> Finally, it runs K-means clustering within each segment to find finer sub-classes.
> The result is a clean separation of sunlit leaves, shadowed leaves, soil, and background."*

**K-Means**
> *"A classic: PCA for dimensionality reduction, then K-means.
> Useful when you want a pure data-driven grouping without any prior assumptions about vegetation indices."*

**SAM — Spectral Angle Mapping**
> *"SAM measures the angle between two spectral vectors rather than their Euclidean distance.
> This makes it illumination-invariant — the magnitude of the reflectance doesn't matter, only the shape of the spectrum.
> Very useful for images with strong shadow gradients."*

**Autoencoder**
> *"A small MLP autoencoder compresses each spectrum down to a low-dimensional latent space,
> then K-means clusters in that space.
> This can capture non-linear structure that PCA misses, but it requires PyTorch and takes longer."*

> Select **Hybrid** for the demo.

---

### Step 4: Number of Classes

*"Use the slider to set how many classes you want.
For hybrid, this controls how many sub-clusters are created within the vegetation and soil segments.
Six is a reasonable default for most crop images."*

---

### Step 5: Parameters (Hybrid)

*"For the Hybrid method, two thresholds matter:"*

- **NDVI threshold (default 0.15)** — Any pixel with NDVI above this is considered vegetation. Lower values include more marginal vegetation; higher values are more conservative.
- **Brightness threshold (default 0.08)** — Pixels with mean reflectance below this value are classified as shadow. Adjust based on your lighting conditions.

*"These two parameters are the main tuning knobs. Everything else is automatic."*

---

### Step 6: Run and Review Results

> Click **Run Analysis**.

*"While it's running you can see live log messages at the bottom — loading time, classify time, quality metrics."*

**When complete:**

*"The banner shows total elapsed time, which is useful for estimating how long a full batch will take."*

*"An HTML report is saved to the output folder. Let me open it."*

> Open the `.html` file in the browser.

**Walk through the report:**

1. **Image Overview** — RGB composite, CIR false-color (NIR-Red-Green), and the full classification map side by side.
2. **Per-Class Images** — Each class rendered individually, highlighted in its color against a darkened background. This makes it easy to visually verify whether the segmentation makes sense.
3. **Classification Summary** — Table showing pixel count and percentage for each class.
4. **Reflectance Spectra** — Interactive Plotly chart. You can hover, zoom, and toggle classes on/off. The shaded bands show ±1 standard deviation.
5. **Quality Assessment** — Silhouette score and Davies-Bouldin index converted to a 0–100% quality gauge.
6. **Vegetation Separation** — NDVI-based recall, precision, and F1 score. This tells you how well the algorithm's leaf classes match the ground-truth vegetation mask defined by NDVI > 0.15.

---

## 5. Output Files (30 seconds)

```
output/
└── image_filename/
    ├── spectra.csv            ← mean & std reflectance per class, per band
    ├── class_map.png          ← classification map with legend
    └── report_20240407_143022.html
```

*"`spectra.csv` is the main data output — one row per band, one column per class.
You can load this directly into Python, R, or Excel for further analysis."*

---

## 6. Key Points to Emphasize (Q&A preparation)

| Question they might ask | Your answer |
|-------------------------|-------------|
| "Does it need GPU?" | No — runs entirely on CPU. Hybrid, K-Means, and SAM use scikit-learn. Autoencoder uses PyTorch on CPU. |
| "How long does it take?" | Depends on image size. A 500×500×200 image takes roughly 30–60 seconds with Hybrid. The elapsed time shown after each run gives you a good estimate. |
| "Can we use our own labels?" | The full version (`app_en.py`) supports Random Forest and 1D-CNN with a labelled pixel CSV. This demo uses the unsupervised-only version. |
| "What if NDVI doesn't work for our crop?" | Adjust the NDVI threshold, or switch to K-Means / SAM which don't rely on vegetation indices. |
| "What wavelength range do you need?" | The core algorithm works with any range, but NDVI requires a Red (~660 nm) and NIR (~800 nm) band. Hybrid automatically falls back to K-Means if those bands are absent. |
| "Can it handle multiple files at once?" | Yes — switch to Batch mode. Each file gets its own report. |

---

## 7. Closing (30 seconds)

*"To summarize: this tool takes a raw hyperspectral field image, automatically segments it into
vegetation, shadow, soil, and background classes without any training data,
and gives you per-class spectra and a quality-assessed HTML report.
The entire pipeline is open-source and runs on a standard lab workstation.
Happy to share the GitHub link and walk through the installation if anyone wants to try it."*

> GitHub: `https://github.com/hoonsoolee/Extract_spectra`

---

*Tip: Before the presentation, do one test run on a real image so you can show an already-generated report without waiting during the demo.*
