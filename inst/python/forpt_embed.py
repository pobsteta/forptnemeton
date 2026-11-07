"""Extraction d'embeddings ForPT (encodeur LitePT-S pre-entraine) pour forptnemeton.

Appele depuis R par reticulate. Reutilise les transformations et le constructeur
de modeles du depot ForPT (prs-eth/ForPT, MIT), qui doit etre importable
(`pip install -e .` dans un clone, ou PYTHONPATH). Le pretraitement reproduit
celui des configurations ForPT : CenterShift(apply_z=True), GridSample a 5 cm,
caracteristiques = coordonnees (in_channels = 3). L'embedding d'une cellule est
la moyenne des caracteristiques de sortie de l'encodeur (504 dimensions), comme
dans la tete de regression d'age (DefaultReggressor).

LitePT exige flash_attn, donc un GPU CUDA.
"""

import numpy as np

# Hyperparametres de l'encodeur, copies de configs/00_ssl_pretrain/pretrain-forpt.py
# (sans masque ni drop path : inference).
BACKBONE = dict(
    type="LitePT-v1m2",
    in_channels=3,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(36, 72, 144, 252, 504),
    enc_num_head=(2, 4, 8, 14, 28),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    enc_conv=(True, True, True, False, False),
    enc_attn=(False, False, False, True, True),
    enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.0,
    shuffle_orders=False,
    pre_norm=True,
    traceable=False,
    enc_mode=True,
    mask_token=False,
)
EMBED_DIM = 504
GRID_SIZE = 0.05
PREFIX = "module.student.backbone."


def environment():
    """Ce qui manque pour executer l'encodeur (liste vide si tout est la)."""
    missing = []
    for mod in ("torch", "spconv", "torch_scatter", "flash_attn", "forpt"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
    if "torch" not in missing:
        import torch
        if not torch.cuda.is_available():
            missing.append("cuda")
    return missing


def load_backbone(weights_path, device="cuda"):
    """Construit LitePT-S et charge les poids pre-entraines (cle student.backbone)."""
    import torch
    import forpt.models  # noqa: F401  (enregistre les modeles)
    from forpt.models import build_model

    model = build_model(dict(BACKBONE))
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    weight = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
    if not weight:
        raise ValueError("No '%s*' key in %s: not a ForPT pretraining checkpoint" % (PREFIX, weights_path))
    res = model.load_state_dict(weight, strict=False)
    # mask_token n'existe qu'au pre-entrainement : seule cle attendue en trop.
    unexpected = [k for k in res.unexpected_keys if "mask_token" not in k]
    if res.missing_keys or unexpected:
        raise ValueError("Checkpoint mismatch: missing %s, unexpected %s" % (res.missing_keys[:5], unexpected[:5]))
    model.to(device).eval()
    return model


def _transform():
    from forpt.datasets.transform import Compose
    return Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=GRID_SIZE, hash_type="fnv", mode="train",
             return_grid_coord=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord"), feat_keys=["coord"]),
    ])


def embed_cells(model, cells, batch_size=16, device="cuda"):
    """Embeddings moyens (n_cellules x 504) d'une liste de nuages (tableaux n x 3).

    Chaque nuage est en metres (X, Y, hauteur normalisee) ; il est recentre par
    CenterShift comme a l'entrainement.
    """
    import torch
    import torch_scatter
    from forpt.datasets.utils import point_collate_fn
    from forpt.models.utils.structure import Point

    transform = _transform()
    out = np.full((len(cells), EMBED_DIM), np.nan, dtype=np.float32)
    for start in range(0, len(cells), batch_size):
        chunk = cells[start:start + batch_size]
        samples = [transform(dict(coord=np.asarray(c, dtype=np.float32).copy())) for c in chunk]
        batch = point_collate_fn(samples)
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        batch["grid_size"] = GRID_SIZE
        with torch.no_grad():
            point = model(Point(batch))
            feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=torch.nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
        out[start:start + len(chunk)] = feat.float().cpu().numpy()
    return out


# --------------------------------------------------------------------------
# Tete d'age (FORage) : modele affine ForPT, un arbre par echantillon.

AGE_WEIGHTS = "04_forage/age-forpt-ft-p100/model/model_best.pth"
AGE_MAX_POINTS = 8192


def load_age_model(weights_path, device="cuda"):
    """DefaultReggressor (LitePT-S + tete MLP) charge depuis model_best.pth."""
    import torch
    import forpt.models  # noqa: F401
    from forpt.models import build_model

    cfg = dict(type="DefaultReggressor", num_targets=1, backbone_embed_dim=EMBED_DIM,
               criteria=None, backbone=dict(BACKBONE))
    model = build_model(cfg)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    weight = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(weight, strict=True)
    model.to(device).eval()
    return model


def predict_ages(model, trees, batch_size=16, device="cuda"):
    """Age predit (annees) pour une liste de nuages d'arbres (tableaux n x 3).

    Le pretraitement de la configuration FORage ne recentre pas : les nuages
    sont passes tels que les donne R, dans le repere de FORage (minimum a 0
    sur chaque axe).
    """
    import torch
    from forpt.datasets.transform import Compose
    from forpt.datasets.utils import point_collate_fn

    transform = Compose([
        dict(type="GridSample", grid_size=GRID_SIZE, hash_type="fnv", mode="train",
             return_grid_coord=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord"), feat_keys=["coord"]),
    ])
    rng = np.random.default_rng(0)
    out = np.full(len(trees), np.nan, dtype=np.float32)
    for start in range(0, len(trees), batch_size):
        chunk = trees[start:start + batch_size]
        samples = []
        for t in chunk:
            c = np.asarray(t, dtype=np.float32).copy()
            if len(c) > AGE_MAX_POINTS:  # FORage : 8192 points par arbre
                c = c[rng.choice(len(c), AGE_MAX_POINTS, replace=False)]
            samples.append(transform(dict(coord=c)))
        batch = point_collate_fn(samples)
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            pred = model(batch)["reg_logits"]
        out[start:start + len(chunk)] = pred.reshape(-1).float().cpu().numpy()
    return out
