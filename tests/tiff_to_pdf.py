from PIL import Image
import glob
import os
import math

# =========================
# CONFIG — edit these
# =========================
input_folder = r"E:\FAU\ss 2025\BIMAP\deepd3\test\test_images\inference\base"
output_pdf   = os.path.join(input_folder, "all_tiffs_one_page.pdf")

# Layout
columns      = 3              # number of columns in the grid
paper_size   = "A4"           # "A4" or "LETTER"
orientation  = "portrait"     # "portrait" or "landscape"
dpi          = 300            # effective layout DPI for page rasterization
margin_mm    = 10             # outer margin around the page
gutter_mm    = 4              # spacing between tiles

# If you want a quick horizontal strip instead of a grid, set columns to a large number.
# =========================

# --- Helpers ---
def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))

def page_pixels(paper, orientation, dpi):
    if paper.upper() == "A4":
        w_in, h_in = 8.27, 11.69
    elif paper.upper() == "LETTER":
        w_in, h_in = 8.5, 11.0
    else:
        raise ValueError("Unsupported paper_size. Use 'A4' or 'LETTER'.")

    if orientation.lower() == "landscape":
        w_in, h_in = h_in, w_in

    return int(round(w_in * dpi)), int(round(h_in * dpi))

# --- Collect TIFFs ---
tiff_files = sorted(
    glob.glob(os.path.join(input_folder, "*.tif")) +
    glob.glob(os.path.join(input_folder, "*.tiff"))
)
if not tiff_files:
    raise SystemExit(f"No TIFFs found in: {input_folder}")

# --- Page + layout metrics ---
page_w, page_h = page_pixels(paper_size, orientation, dpi)
margin = mm_to_px(margin_mm, dpi)
gutter = mm_to_px(gutter_mm, dpi)

N = len(tiff_files)
cols = max(1, int(columns))
rows = math.ceil(N / cols)

inner_w = page_w - 2*margin - (cols - 1)*gutter
inner_h = page_h - 2*margin - (rows - 1)*gutter
if inner_w <= 0 or inner_h <= 0:
    raise SystemExit("Margins/gutters too large for page size at this DPI.")

cell_w = inner_w // cols
cell_h = inner_h // rows
if cell_w <= 0 or cell_h <= 0:
    raise SystemExit("Cells ended up zero or negative sized; reduce columns or margins/gutters, or increase DPI.")

# --- Create page canvas ---
page = Image.new("RGB", (page_w, page_h), "white")

# --- Paste images into grid ---
x0, y0 = margin, margin
idx = 0
for r in range(rows):
    for c in range(cols):
        if idx >= N:
            break

        path = tiff_files[idx]
        try:
            im = Image.open(path)
            # If multi-frame TIFF, use first frame for a contact sheet
            try:
                im.seek(0)
            except Exception:
                pass

            # Convert to RGB (PDF can't handle some modes directly)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")

            # Fit into cell preserving aspect ratio
            w, h = im.size
            scale = min(cell_w / w, cell_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)

            # Center inside the cell
            cell_x = x0 + c * (cell_w + gutter)
            cell_y = y0 + r * (cell_h + gutter)
            paste_x = cell_x + (cell_w - new_w) // 2
            paste_y = cell_y + (cell_h - new_h) // 2

            page.paste(im, (paste_x, paste_y))
        except Exception as e:
            print(f"Skipping {os.path.basename(path)}: {e}")

        idx += 1

# --- Save as single-page PDF ---
# The PDF page will match the pixel dimensions; it’s a single page containing the collage.
page.save(output_pdf, "PDF", resolution=dpi)
print(f"✅ Saved single-page collage PDF:\n{output_pdf}")
