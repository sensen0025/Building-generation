"""Compose a panorama seed: a Minecraft building on grass under a sky with a real horizon.

v6 lesson, kept as the reason this file exists: HY-Pano is image-conditioned, so the seed dominates.
The v6 seed put the building across the whole frame AND on a transparent background; the pano model
read that as "the building surrounds the camera everywhere" and produced two curved facades at the
left/right edges with a copy in the far centre - and the generated cloud had no coherent building.

So this composition deliberately does the opposite of "fill the frame":
  * the building occupies --subject-width of the canvas (~0.5), centred,
  * the horizon sits LOW (--horizon 0.78) so ground reads as a large plane and the sky is a band,
  * the building's feet stand ON the horizon line, so ground and sky meet behind it,
  * a 3/4 render (two facades visible) is used as the source, giving the pano more known surface.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def subject_alpha(img):
    if img.mode in ('RGBA', 'LA'):
        return np.asarray(img.convert('RGBA'))[:, :, 3]
    a = np.asarray(img.convert('RGB')).astype(np.int16)
    mx, mn = a.max(2), a.min(2)
    return (~((mn > 235) & ((mx - mn) < 12))).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=Path, required=True, help='3/4 render with alpha')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--width', type=int, default=1600)
    ap.add_argument('--height', type=int, default=900)
    ap.add_argument('--subject-width', type=float, default=0.50, help='building width / canvas width')
    ap.add_argument('--horizon', type=float, default=0.78, help='horizon y / canvas height')
    ap.add_argument('--shadow', type=float, default=0.18, help='contact shadow strength')
    args = ap.parse_args()

    src = Image.open(args.image)
    alpha = subject_alpha(src)
    ys, xs = np.nonzero(alpha > 128)
    if not len(ys):
        raise SystemExit('no opaque subject found')
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    subj = src.convert('RGBA').crop(box)
    alpha = alpha[box[1]:box[3], box[0]:box[2]]

    W, H = args.width, args.height
    scale = (W * args.subject_width) / subj.width
    nw, nh = int(subj.width * scale), int(subj.height * scale)
    subj = subj.resize((nw, nh), Image.LANCZOS)
    alpha_img = Image.fromarray(alpha).resize((nw, nh), Image.LANCZOS)

    horizon_y = int(H * args.horizon)
    feet = int(np.nonzero(np.asarray(alpha_img) > 128)[0].max())
    px, py = (W - nw) // 2, horizon_y - feet
    print(f'subject {subj.size}  feet {feet}  horizon {horizon_y}  paste ({px},{py})')

    # sky: daylight gradient; ground: grass that darkens towards the camera
    canvas = np.zeros((H, W, 3), np.float32)
    sky_top, sky_bot = np.array([116, 170, 224], np.float32), np.array([196, 220, 240], np.float32)
    grass_far, grass_near = np.array([128, 172, 86], np.float32), np.array([96, 138, 62], np.float32)
    for y in range(H):
        t = min(y / max(horizon_y - 1, 1), 1.0)
        if y < horizon_y:
            canvas[y] = sky_top * (1 - t) + sky_bot * t
        else:
            u = (y - horizon_y) / max(H - horizon_y - 1, 1)
            canvas[y] = grass_far * (1 - u) + grass_near * u
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

    # faint grass-block checker, fading with distance
    d = ImageDraw.Draw(img, 'RGBA')
    step = max(H // 40, 8)
    for gi, y in enumerate(range(horizon_y, H, step)):
        fade = int(18 * gi / max((H - horizon_y) // step, 1))
        for x in range(0, W, step):
            if ((x // step) + gi) % 2 == 0:
                d.rectangle([x, y, x + step, y + step], fill=(255, 255, 255, fade))

    # contact shadow under the building so it sits on the ground
    sh = alpha_img.filter(ImageFilter.GaussianBlur(step * 0.8))
    sh_arr = (np.asarray(sh).astype(np.float32) / 255.0 * args.shadow * 255).astype(np.uint8)
    shadow = Image.new('RGBA', (nw, nh), (0, 0, 0, 0))
    shadow.putalpha(Image.fromarray(sh_arr))
    img.paste(shadow, (px, py + int(step * 0.35)), shadow)

    img.paste(subj, (px, py), alpha_img)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print('SAVED', args.out, img.size)


if __name__ == '__main__':
    main()
