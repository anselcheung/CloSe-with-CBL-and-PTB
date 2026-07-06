import warnings
import sys
import yaml

from lib.utils.types import EasierDict
from lib.utils.config import load_model
from lib import factory

warnings.filterwarnings('ignore')

# ─── Config notes ─────────────────────────────────────────────────────────────
# This script reads a YAML config (default: cfg/closenet_test.yaml) and a
# pretrained model checkpoint (default: pretrained/closenet.pth).
#
# Before running, verify these fields in the YAML config:
#
#   data.data_path    — root folder containing the .npz scan files
#                       (e.g. ./data — the folder downloaded from HuggingFace)
#   data.split_file   — path to the split JSON/NPZ defining train/val/test
#                       (default: ./cfg/data_split.json — 162 test samples)
#   data.batch_size   — inference batch size; reduce if you run out of GPU memory
#   data.num_workers  — DataLoader workers (0 = main process only, safe default)
#   device            — 'cuda' or 'cpu'
#
#   exp_logs_path     — IMPORTANT: set this to a directory that does NOT contain
#                       any prior training checkpoints (.pt files). The trainer
#                       will auto-load the latest .pt from
#                       <exp_logs_path>/checkpoints/ and overwrite the pretrained
#                       weights if it finds one. Point this at a fresh/empty dir
#                       (e.g. ./closenet_test_run) to avoid that.
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg_path   = sys.argv[1] if len(sys.argv) > 1 else './cfg/closenet_test.yaml'
    model_path = sys.argv[2] if len(sys.argv) > 2 else './pretrained/closenet.pth'

    cfg = EasierDict(yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader))

    # load_model() reads the .pth weights AND the paired <name>_cfg.yaml for
    # the model architecture, so no separate arch config is needed here.
    model = load_model(model_path, device=cfg.device)

    # Only instantiate the test split — train and val are not loaded.
    test_data = factory.get_dataset(cfg, 'test')

    # Pass None for unused train/val; the trainer only touches them during
    # train_model(), which we do not call here.
    trainer = factory.get_trainer(model, None, None, test_data, cfg)

    print('Running inference on test set...')
    metrics, _ = trainer.evaluate_model(dataset='test', seed=42)

    print('\n=== Test Set Results ===')
    print(f"  mIoU:     {metrics['mIoU']:.4f}")
    print(f"  freq_IoU: {metrics['freq_IoU']:.4f}")
    print(f"  IoU (per class): {metrics['IoU']}")
