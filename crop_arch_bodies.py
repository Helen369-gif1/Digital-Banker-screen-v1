import json
from PIL import Image

with open("arch_transforms_new.json", encoding="utf-8") as f:
    data = json.load(f)

arches = data["arches"]

for n, info in arches.items():
    bbox = info["main_arch_bbox"]
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    src_path = f"cropped-new/{n}.png"
    im = Image.open(src_path).convert("RGBA")
    body = im.crop((x, y, x + w, y + h))
    out_path = f"arch-body-{n}.png"
    body.save(out_path)
    print(f"{n}: cropped {src_path} bbox={bbox} -> {out_path} size={body.size}")

# --- derive a shared hit-test silhouette polygon from arch-body-1 ---
# All six bodies share the same w/h and are the same asset, so one
# alpha-derived polygon is reused for every .arch-hit clip-path.
im = Image.open("arch-body-1.png").convert("RGBA")
W, H = im.size
alpha = im.split()[-1]
px = alpha.load()

SAMPLES = 12
ALPHA_THRESHOLD = 20

left_pts = []
right_pts = []
for i in range(SAMPLES + 1):
    y = min(H - 1, round(i * H / SAMPLES))
    xs = [x for x in range(W) if px[x, y] >= ALPHA_THRESHOLD]
    if not xs:
        continue
    x0, x1 = min(xs), max(xs)
    left_pts.append((x0 / W * 100, y / H * 100))
    right_pts.append((x1 / W * 100, y / H * 100))

# Build polygon: down the left edge, then back up the right edge.
poly_pts = left_pts + list(reversed(right_pts))
poly_str = ", ".join(f"{px_:.2f}% {py_:.2f}%" for px_, py_ in poly_pts)
print("\nCLIP-PATH POLYGON (shared, from arch-body-1):")
print(f"clip-path: polygon({poly_str});")
