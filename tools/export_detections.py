import os
import cv2
import torch
import numpy as np

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

#OUTPUT FORMAT: frame_id, x1, y1, x2, y2, detection_confidence_score
# detection_confidence_score = score = obj_conf × class_conf



# ---------------- CONFIG ----------------
EXP_FILE = "exps/drosophila_yolox_s.py"
CKPT = "/home/sampsonj2/Desktop/YOLOX_outputs/yolox_base/best_ckpt.pth"

IMG_DIR = "/home/sampsonj2/Desktop/datasets/drosophila/val2017/006"
OUT_TXT = "/home/sampsonj2/Desktop/detections_006.txt"

CONF_THRES = 0.03
NMS_THRES = 0.6
TEST_SIZE = (960, 960)
DEVICE = "cpu"   # or "cuda"
# ----------------------------------------


def compute_ratio(h: int, w: int, test_size):
    """Match YOLOX resize behavior: ratio = min(test_h/h, test_w/w). Always float."""
    test_h, test_w = test_size
    return float(min(test_h / float(h), test_w / float(w)))


def main():
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    model.eval()

    device = torch.device(DEVICE)
    model.to(device)

    ckpt = torch.load(CKPT, map_location="cpu")
    # YOLOX checkpoints often have {"model": OrderedDict(...)}
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)

    preproc = ValTransform(legacy=False)

    image_files = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    frame_id = 1
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)

    with open(OUT_TXT, "w") as out:
        for name in image_files:
            img_path = os.path.join(IMG_DIR, name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] could not read: {img_path}")
                frame_id += 1
                continue

            h, w = img.shape[:2]
            ratio = compute_ratio(h, w, TEST_SIZE)

            img_t, _ = preproc(img, None, TEST_SIZE)  # ignore returned ratio; compute our own
            img_t = torch.from_numpy(img_t).unsqueeze(0).to(device)
            img_t = img_t.float()

            with torch.no_grad():
                outputs = model(img_t)
                outputs = postprocess(
                    outputs,
                    num_classes=exp.num_classes,
                    conf_thre=CONF_THRES,
                    nms_thre=NMS_THRES,
                )

            dets = outputs[0]
            if dets is not None:
                # dets is Nx7: x1,y1,x2,y2,obj_conf,cls_conf,cls_id
                dets = dets.detach().cpu().numpy()
                for det in dets:
                    x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det[:7]
                    score = float(obj_conf * cls_conf)

                    # scale coords back to original image space
                    x1 = float(x1 / ratio)
                    y1 = float(y1 / ratio)
                    x2 = float(x2 / ratio)
                    y2 = float(y2 / ratio)

                    out.write(f"{frame_id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{score:.6f}\n")

            frame_id += 1

    print("Saved detections to:", OUT_TXT)


if __name__ == "__main__":
    main()

