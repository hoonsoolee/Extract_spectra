from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FRAME_DIR = ROOT / "demo_frames"
OUTPUT = ROOT / "hyperspectral_pipeline_live_demo_en.mp4"

WIDTH = 1600
HEIGHT = 900
FPS = 30
CAPTION_HEIGHT = 118
CONTENT_HEIGHT = HEIGHT - CAPTION_HEIGHT
BACKGROUND = (244, 247, 249)
NAVY = (21, 39, 56)
GREEN = (44, 134, 83)


@dataclass(frozen=True)
class Scene:
    image_name: str | None
    title: str
    subtitle: str
    seconds: float


SCENES = [
    Scene(
        None,
        "Hyperspectral Crop Analysis Pipeline",
        "Recorded from a live run · AP3-4.bil · K-means · 6 clusters",
        4.0,
    ),
    Scene(
        "01_home.png",
        "1. Open the English web application",
        "The whole-field workflow, ROI spectra, and calibration tools share one interface.",
        4.0,
    ),
    Scene(
        "02_data_selected.png",
        "2. Select the data and analysis settings",
        "Choose a local folder, one file or a batch, cluster count, report type, and downsampling.",
        5.0,
    ),
    Scene(
        "04_analysis_complete.png",
        "3. Run whole-field clustering",
        "This live sample finished in about 10 seconds at downsample ×8.",
        7.0,
    ),
    Scene(
        "04_analysis_complete.png",
        "4. Visually inspect clustering quality",
        "Compare RGB, the cluster color map, and the adjustable RGB overlay.",
        7.0,
    ),
    Scene(
        "05_report_actions.png",
        "5. Review cluster proportions and export results",
        "Open the HTML report or the result folder, or download the report directly.",
        5.0,
    ),
    Scene(
        "06_roi_load_screen.png",
        "6. Load an image for targeted ROI analysis",
        "The same BIL/ENVI file list is available without typing a long path.",
        4.0,
    ),
    Scene(
        "09_roi_spectrum.png",
        "7. Draw a box, lasso, or polygon ROI",
        "Here, a box ROI was dragged directly over the crop canopy.",
        5.0,
    ),
    Scene(
        "11_roi_graph_full.png",
        "8. Inspect and export the ROI spectrum",
        "Mean, median, standard deviation, wavelength range, and QC indicators are shown.",
        7.0,
    ),
    Scene(
        "12_panel_calibration.png",
        "9. Build a validated white/dark calibration",
        "Register panel reflectance and dark reference values; failed QC is blocked automatically.",
        5.0,
    ),
    Scene(
        None,
        "Field data → science-ready outputs",
        "HTML report · CSV/XLSX · RGB/NDVI · cluster maps · raw and calibrated spectra",
        5.0,
    ),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    candidates = [Path("C:/Windows/Fonts") / name, Path(name)]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


TITLE_FONT = _font(34, bold=True)
SUBTITLE_FONT = _font(22)
CARD_TITLE_FONT = _font(58, bold=True)
CARD_SUBTITLE_FONT = _font(30)
SMALL_FONT = _font(20)


def _fit_image(source: Image.Image) -> Image.Image:
    source = source.convert("RGB")
    margin = 28
    max_w = WIDTH - 2 * margin
    max_h = CONTENT_HEIGHT - 2 * margin
    scale = min(max_w / source.width, max_h / source.height)
    size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    resized = source.resize(size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (WIDTH, CONTENT_HEIGHT), BACKGROUND)
    x = (WIDTH - size[0]) // 2
    y = (CONTENT_HEIGHT - size[1]) // 2
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (x + 7, y + 9, x + size[0] + 7, y + size[1] + 9),
        radius=12,
        fill=(24, 39, 55, 42),
    )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow).convert("RGB")
    canvas.paste(resized, (x, y))
    return canvas


def _title_card(title: str, subtitle: str) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((112, 108, 156, 700), radius=20, fill=GREEN)
    draw.text((205, 275), title, font=CARD_TITLE_FONT, fill="white")
    draw.text((207, 370), subtitle, font=CARD_SUBTITLE_FONT, fill=(207, 223, 231))
    draw.text(
        (207, 705),
        "UIUC field hyperspectral workflow · English demonstration",
        font=SMALL_FONT,
        fill=(152, 178, 191),
    )
    return canvas


def _captioned(scene: Scene) -> Image.Image:
    if scene.image_name is None:
        return _title_card(scene.title, scene.subtitle)

    source_path = FRAME_DIR / scene.image_name
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    content = _fit_image(Image.open(source_path))
    canvas = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    canvas.paste(content, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, CONTENT_HEIGHT, 12, HEIGHT), fill=GREEN)
    draw.text((42, CONTENT_HEIGHT + 18), scene.title, font=TITLE_FONT, fill="white")
    draw.text(
        (44, CONTENT_HEIGHT + 65),
        scene.subtitle,
        font=SUBTITLE_FONT,
        fill=(207, 223, 231),
    )
    return canvas


def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def build_video() -> Path:
    frames = [_to_bgr(_captioned(scene)) for scene in SCENES]
    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not initialize the MP4 writer")

    transition_frames = int(0.45 * FPS)
    try:
        for index, (scene, frame) in enumerate(zip(SCENES, frames)):
            hold_frames = max(1, int(scene.seconds * FPS) - transition_frames)
            for _ in range(hold_frames):
                writer.write(frame)

            if index + 1 < len(frames):
                next_frame = frames[index + 1]
                for step in range(transition_frames):
                    alpha = (step + 1) / transition_frames
                    writer.write(cv2.addWeighted(frame, 1.0 - alpha, next_frame, alpha, 0))
    finally:
        writer.release()

    return OUTPUT


if __name__ == "__main__":
    result = build_video()
    print(result)
