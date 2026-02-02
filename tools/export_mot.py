#!/usr/bin/env python3
import os
import glob
import argparse

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess


def get_image_list(path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if os.path.isdir(path):
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, f"*{e}")))
        return sorted(files)
    else:
        return [path]


def preprocess(img, test_size):
    transform = ValTransform(legacy=False)
    img, _ = transform(img, None, test_size)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_file", "-f", required=True)
    ap.add_argument("--ckpt", "-c", required=True)
    ap.add_argument("--path", required=True, help="image file or directory of frames")
    ap.add_argument("--out", required=True, help="output txt file")
    ap.add_argument("--conf", type=float, default=0.03)
    ap.add_argument("--nms", type=float, default=0.6)
    ap.add_argument("--tsize", type=int, default=960)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--class_agnostic", action="store_true")
    args = ap.parse_args()

    exp = get_exp(args.exp_file, None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    exp.test_size = (args.tsize, args.tsize)

    device = torch.device(args.device)

    model = exp.get_model()
    model.to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    image_paths = get_image_list(args.path)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {args.path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w") as f:
        for frame_idx, p in enumerate(image_paths, start=1):
            img0 = cv2.imread(p)
            if img0 is None:
                print(f"[WARN] could not read: {p}")
                continue

            h0, w0 = img0.shape[:2]
            inp = preprocess(img0, exp.test_size).to(device)

            with torch.no_grad():
                outputs = model(inp)
                outputs = postprocess(
                    outputs,
                    num_classes=exp.num_classes,
                    conf_thre=exp.test_conf,
                    nms_thre=exp.nmsthre,
                    class_agnostic=args.class_agnostic,
                )

            det = outputs[0]
            if det is None:
                continue

            det = det.cpu()

            # scale back from resized coords to original pixel coords
            r = min(exp.test_size[0] / h0, exp.test_size[1] / w0)

            x1 = det[:, 0] / r
            y1 = det[:, 1] / r
            x2 = det[:, 2] / r
            y2 = det[:, 3] / r

            obj_conf = det[:, 4]
            cls_conf = det[:, 5]
            score = obj_conf * cls_conf

            for i in range(det.shape[0]):
                f.write(
                    f"{frame_idx},{x1[i].item():.2f},{y1[i].item():.2f},{x2[i].item():.2f},{y2[i].item():.2f},{score[i].item():.6f}\n"
                )

    print(f"Saved detections to: {args.out}")
    print(f"Frames processed: {len(image_paths)}")


if __name__ == "__main__":
    main()
