"""
config.py
=========
All argument parsing, global constants, class maps, prompts, and
path/hyperparameter configuration for the GastroVision pipeline.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GastroVision Augmentation Pipeline")

    # Paths
    p.add_argument("--data_dir",       default="/data")
    p.add_argument("--output_dir",     default="/output")
    p.add_argument("--image_root",     default="gastrovision_raw/Gastrovision")
    p.add_argument("--train_csv",      default="train.csv")
    p.add_argument("--val_csv",        default="val.csv")
    p.add_argument("--test_csv",       default="test.csv")
    p.add_argument("--aug_train_csv",  default="train_aug.csv")
    p.add_argument("--synth_csv",      default="synthetic_train.csv")
    p.add_argument("--synth_dir",      default="synthetic")

    # Classifier hyperparameters
    p.add_argument("--img_size",         type=int,   default=224)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--freeze_epochs",    type=int,   default=16)
    p.add_argument("--fine_tune_epochs", type=int,   default=24)
    p.add_argument("--gamma",            type=float, default=2.0)
    p.add_argument("--freeze_lr_mult",   type=float, default=10.0)

    # Diffusion / SD config
    p.add_argument("--sd_model_id",         default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora_rank",           type=int,   default=32)
    p.add_argument("--lora_alpha",          type=int,   default=64)
    p.add_argument("--lora_dropout",        type=float, default=0.1)
    p.add_argument("--domain_adapt_steps",  type=int,   default=15000)
    p.add_argument("--sd_batch_size",       type=int,   default=4)
    p.add_argument("--sd_grad_accum",       type=int,   default=4)
    p.add_argument("--sd_lr",              type=float, default=1e-4)
    p.add_argument("--ema_decay",           type=float, default=0.9999)
    p.add_argument("--ema_warmup_steps",    type=int,   default=100)
    p.add_argument("--samples_per_class",   type=int,   default=500)
    p.add_argument("--gen_steps",           type=int,   default=50)
    p.add_argument("--guidance_scale",      type=float, default=7.5)
    p.add_argument("--gen_batch_size",      type=int,   default=4)

    # Evaluation & ablation
    p.add_argument("--kfold_splits",            type=int,   default=5)
    p.add_argument("--n_seeds",                 type=int,   default=3,
                   help="Number of seeds for stability analysis")
    p.add_argument("--min_reliable_samples",    type=int,   default=10)
    p.add_argument("--rare_threshold",          type=int,   default=30,
                   help="Classes with fewer samples are considered rare")
    p.add_argument("--ultraRare_threshold",     type=int,   default=15,
                   help="Classes with fewer samples use few-shot track")
    p.add_argument("--seed",                    type=int,   default=42)
    p.add_argument("--min_free_disk_gb",        type=float, default=20.0)

    # Execution modes
    p.add_argument("--skip_domain_adapt",   action="store_true")
    p.add_argument("--skip_generation",     action="store_true")
    p.add_argument("--skip_training",       action="store_true")
    p.add_argument("--evaluate_only",       action="store_true")
    p.add_argument("--run_ablations",             action="store_true")
    p.add_argument("--run_reviewer_experiments",  action="store_true",
                   help="Run HybridV2 internal ablation + synthetic-only experiment (reviewer response)")
    p.add_argument("--run_fewshot",               action="store_true")
    p.add_argument("--tune",                action="store_true")
    p.add_argument("--tune_trials",         type=int,   default=15)
    p.add_argument("--tune_epochs",         type=int,   default=8)
    p.add_argument("--models", nargs="+",
                   default=["efficientnetv2_rw_s", "swin_v2", "dinov2",
                            "hybrid_cnn_transformer_v2"],
                   help="Which classifier models to train and evaluate.")

    return p.parse_args()


# ==============================================================================
# Global state (populated after parse_args)
# ==============================================================================

args = parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DATA_DIR        = Path(args.data_dir)
OUTPUT_DIR      = Path(args.output_dir)
IMAGE_ROOT_DIR  = DATA_DIR / args.image_root
SPLITS_DIR      = OUTPUT_DIR / "splits"
SYNTH_DIR       = OUTPUT_DIR / args.synth_dir
CKPT_DIR        = OUTPUT_DIR / "checkpoints"
RESULTS_DIR     = OUTPUT_DIR / "results"
LOGS_DIR        = OUTPUT_DIR / "logs"
GRADCAM_DIR     = RESULTS_DIR / "gradcam"
CALIB_DIR       = RESULTS_DIR / "calibration"
TSNE_DIR        = RESULTS_DIR / "tsne"

for d in [SPLITS_DIR, SYNTH_DIR, CKPT_DIR, RESULTS_DIR, LOGS_DIR,
          GRADCAM_DIR, CALIB_DIR, TSNE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Populated after split creation
NUM_CLASSES   = None
RARE_CLASSES  = []       # contiguous indices, n < rare_threshold
ULTRA_RARE    = []       # contiguous indices, n < ultraRare_threshold (few-shot track)
LABEL_MAP     = {}       # original_label -> contiguous index
REV_LABEL_MAP = {}       # contiguous -> original_label


# ==============================================================================
# Class maps & prompts
# ==============================================================================

CLASS_MAP = {
    "Accessory tools": 0, "Angiectasia": 1,
    "Barretts esophagus": 2,
    "Barrett’s esophagus": 2,
    "Barrett's esophagus": 2,
    "Blood in lumen": 3, "Cecum": 4, "Colon diverticula": 5,
    "Colon polyps": 6, "Colorectal cancer": 7, "Duodenal bulb": 8,
    "Dyed-lifted-polyps": 9, "Dyed-resection-margins": 10, "Erythema": 11,
    "Esophageal varices": 12, "Esophagitis": 13, "Gastric polyps": 14,
    "Gastroesophageal_junction_normal z-line": 15, "Ileocecal valve": 16,
    "Mucosal inflammation large bowel": 17, "Normal esophagus": 18,
    "Normal mucosa and vascular pattern in the large bowel": 19,
    "Normal stomach": 20, "Pylorus": 21, "Resected polyps": 22,
    "Resection margins": 23, "Retroflex rectum": 24,
    "Small bowel_terminal ileum": 25, "Ulcer": 26,
}

CLASS_NAMES = [
    "Accessory tools", "Angiectasia", "Barrett's esophagus", "Blood in lumen",
    "Cecum", "Colon diverticula", "Colon polyps", "Colorectal cancer",
    "Duodenal bulb", "Dyed-lifted-polyps", "Dyed-resection-margins", "Erythema",
    "Esophageal varices", "Esophagitis", "Gastric polyps",
    "Gastroesophageal junction (z-line)", "Ileocecal valve",
    "Mucosal inflammation LB", "Normal esophagus",
    "Normal mucosa LB", "Normal stomach", "Pylorus",
    "Resected polyps", "Resection margins", "Retroflex rectum",
    "Small bowel/terminal ileum", "Ulcer",
]

# Sample counts per class (from actual dataset)
CLASS_COUNTS = {
    0: 1266, 1: 17, 2: 95, 3: 171, 4: 113, 5: 29, 6: 820, 7: 139,
    8: 205, 9: 141, 10: 246, 11: 15, 12: 7, 13: 107, 14: 65,
    15: 330, 16: 200, 17: 29, 18: 140, 19: 1467, 20: 969, 21: 393,
    22: 92, 23: 25, 24: 67, 25: 846, 26: 6,
}

DOMAIN_PREFIX = "endoscopy photo, circular vignette, specular highlights, pink mucosa: "

NEGATIVE_PROMPT = (
    "illustration, diagram, cartoon, drawing, text, watermark, "
    "x-ray, mri, ct scan, histology, microscopy, "
    "blurry, low quality, overexposed, noisy, "
    "natural scene, person, face, outdoor"
)

CLASS_PROMPTS = {
    0:  "metal endoscopic tools, forceps or snare visible, gastroscopy",
    1:  "angiectasia, tortuous red vessels, salmon mucosa, capsule endoscopy",
    2:  "Barrett's esophagus, salmon irregular patches, lower esophagus",
    3:  "blood in lumen, dark red pooling, gastric cavity",
    4:  "cecum, pale pink mucosa, appendiceal orifice, haustral folds",
    5:  "colon diverticula, dark circular openings in colonic wall",
    6:  "colon polyp, sessile or pedunculated lesion, pink mucosa",
    7:  "colorectal cancer, irregular friable mass, ulceration, colon",
    8:  "duodenal bulb, pale smooth mucosa, circular folds",
    9:  "dyed lifted polyp, blue submucosal injection, raised lesion",
    10: "dyed resection margins, blue mucosal edges, post-polypectomy",
    11: "gastric erythema, diffuse reddish mucosal discoloration, stomach",
    12: "esophageal varices, bluish bulging veins, longitudinal, esophagus",
    13: "esophagitis, erythematous mucosa, linear erosions, esophagus",
    14: "gastric polyp, smooth rounded lesion, gastric wall",
    15: "gastroesophageal junction, z-line, squamocolumnar border",
    16: "ileocecal valve, two lips visible, cecal mucosa",
    17: "mucosal inflammation, granular friable reddish colon, lost vascular pattern",
    18: "normal esophagus, smooth pale pink mucosa, longitudinal folds",
    19: "normal colon, smooth pink mucosa, clear vascular pattern, haustrae",
    20: "normal stomach, rugal folds, pink gastric mucosa, gastric pool",
    21: "pylorus, circular orifice, antral folds, gastroscopy",
    22: "resected polyp, post-polypectomy scar, cauterized flat defect",
    23: "resection margins, cauterized edges, whitish fibrinous border",
    24: "retroflex rectum, retroflexed view, anorectal junction",
    25: "terminal ileum, pale villous mucosa, fine texture, small bowel",
    26: "gastric ulcer, mucosal crater, white fibrinous base, erythematous rim",
}

# FID transform: InceptionV3 expects [0,1] with transform_input=False
FID_TRANSFORM = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
])


# ==============================================================================
# Per-model hyperparameters
# ==============================================================================

HPARAMS = {
    "efficientnetv2_rw_s": {
        "lr": args.lr, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "gamma": args.gamma, "freeze_lr_mult": args.freeze_lr_mult,
        "weight_decay": args.weight_decay,
    },
    "swin_v2": {
        "lr": args.lr, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "gamma": args.gamma, "freeze_lr_mult": args.freeze_lr_mult,
        "weight_decay": args.weight_decay,
    },
    "dinov2": {
        # ViT backbones suffer catastrophic forgetting with high LR on full unfreeze.
        # Use lr*0.05 (1.5e-5) for fine-tune phase; freeze_lr_mult keeps head LR at
        # 5*1.5e-5=7.5e-5 during phase 1 which is safe.
        "lr": args.lr * 0.05, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs + 8,
        "batch_size": min(args.batch_size, 16),
        "gamma": args.gamma, "freeze_lr_mult": 5.0,
        "weight_decay": args.weight_decay,
    },
    "hybrid_cnn_transformer_v2": {
        "lr": args.lr * 0.67, "freeze_epochs": max(1, args.freeze_epochs - 8),
        "fine_tune_epochs": args.fine_tune_epochs,
        "batch_size": min(args.batch_size, 16),
        "gamma": args.gamma, "freeze_lr_mult": 5.0,
        "weight_decay": args.weight_decay,
    },
}

# Cap batch sizes on ≤11GB GPUs
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1e9 < 20:
    for k in HPARAMS:
        HPARAMS[k]["batch_size"] = min(HPARAMS[k]["batch_size"], 16)
    HPARAMS["swin_v2"]["batch_size"] = 8
    HPARAMS["dinov2"]["batch_size"] = 8
    HPARAMS["hybrid_cnn_transformer_v2"]["batch_size"] = 8
    print("  ⚠ ≤11GB VRAM detected — batch sizes capped")
