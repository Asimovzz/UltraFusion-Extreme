import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.test_dataset import TestDataset
from model.raft.raft import RAFT
from ultrafusion_extreme.alignment import choose_alignment, flow_magnitude, summarize_mask
from utils.flow import IMF, backward_warp, forward_backward_consistency_check


def pad_to_multiple(x, multiple):
    _, _, h, w = x.size()
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), "reflect")


def load_raft(ckpt_path, device):
    args = argparse.Namespace(dropout=0, alternate_corr=False)
    model = RAFT(args).to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    clean_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_state_dict[key.replace("module.", "")] = value
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model


def diagnose_pair(ue, oe, flow_model, device, iters, saturation_threshold):
    _, _, h, w = oe.shape
    ue_pad = pad_to_multiple(ue.to(device), 16)
    oe_pad = pad_to_multiple(oe.to(device), 16)

    with torch.no_grad():
        ue_imf = IMF(ue_pad, oe_pad)
        _, flow_ue_to_oe = flow_model(ue_imf * 2 - 1, oe_pad * 2 - 1, iters=iters, test_mode=True)
        _, flow_oe_to_ue = flow_model(oe_pad * 2 - 1, ue_imf * 2 - 1, iters=iters, test_mode=True)
        warped_ue = backward_warp(ue_pad, flow_oe_to_ue)
        _, occ_mask = forward_backward_consistency_check(flow_ue_to_oe, flow_oe_to_ue)
        occ_mask = occ_mask.unsqueeze(1)

    ue_imf = ue_imf[:, :, :h, :w]
    warped_ue = warped_ue[:, :, :h, :w]
    occ_mask = occ_mask[:, :, :h, :w]
    flow_mag = flow_magnitude(flow_oe_to_ue)[:, :, :h, :w]
    stats = summarize_mask(occ_mask, oe.to(device), saturation_threshold)
    stats["flow_mean"] = float(flow_mag.mean().detach().cpu())
    stats["flow_p95"] = float(torch.quantile(flow_mag.flatten().detach().cpu(), 0.95))
    stats["recommended_strategy"] = choose_alignment(stats)

    return {
        "ue_imf": ue_imf,
        "warped_ue": warped_ue,
        "occ_mask": occ_mask,
        "flow_mag": flow_mag,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose UltraFusion pre-alignment on ue/oe image pairs.")
    parser.add_argument("--dataset", default="UltraFusionExtreme")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output", default="reports/alignment_diagnosis")
    parser.add_argument("--raft_ckpt", default="ckpts/raft-sintel.pth")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--saturation_threshold", type=float, default=0.98)
    parser.add_argument("--max_long_edge", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TestDataset(args.dataset, input_dir=args.input_dir, max_long_edge=args.max_long_edge)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0)
    flow_model = load_raft(args.raft_ckpt, args.device)

    report = []
    for batch in dataloader:
        name = batch["file_name"][0]
        scene_dir = output_dir / name
        scene_dir.mkdir(parents=True, exist_ok=True)
        ue = batch["ue"].to(args.device)
        oe = batch["oe"].to(args.device)
        result = diagnose_pair(ue, oe, flow_model, args.device, args.iters, args.saturation_threshold)

        save_image(ue, scene_dir / "ue.png")
        save_image(oe, scene_dir / "oe.png")
        save_image(result["ue_imf"], scene_dir / "ue_imf.png")
        save_image(result["warped_ue"], scene_dir / "warped_ue_to_oe.png")
        save_image(result["occ_mask"], scene_dir / "occ_mask.png")
        save_image(result["flow_mag"] / (result["flow_mag"].max() + 1e-6), scene_dir / "flow_magnitude.png")

        item = {"scene": name, **result["stats"]}
        report.append(item)
        print(json.dumps(item, ensure_ascii=False))

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
