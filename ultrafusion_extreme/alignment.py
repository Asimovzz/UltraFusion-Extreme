import torch


def summarize_mask(occ_mask, oe, saturation_threshold=0.98):
    return {
        "occ_ratio": float(occ_mask.mean().detach().cpu()),
        "saturation_ratio": float((oe >= saturation_threshold).float().mean().detach().cpu()),
    }


def choose_alignment(stats, min_occ_ratio=0.002, max_occ_ratio=0.35, saturation_ratio=0.20):
    unreliable_mask = stats["occ_ratio"] < min_occ_ratio or stats["occ_ratio"] > max_occ_ratio
    saturation_heavy = stats["saturation_ratio"] > saturation_ratio
    return "noalign" if unreliable_mask and saturation_heavy else "align"


def flow_magnitude(flow):
    return torch.norm(flow, dim=1, keepdim=True)

