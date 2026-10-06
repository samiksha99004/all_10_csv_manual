"""Windows side: publish the current model files to the victim bundle.

Copies the 2-level model files and the monitor from the project into
live_detection/victim/ (the folder the victim sees as /media/sf_F_DRIVE/all_10_csv_manual/live_detection/victim),
and writes SHA256SUMS so the victim can copy only what changed.

Run from the project root after any model or monitor change:
  python live_detection/victim/publish_models.py
"""
import hashlib
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLE = os.path.join(ROOT, "live_detection", "victim")
MODELS = os.path.join(BUNDLE, "models")

# source (in the project) -> destination (in the bundle)
SOURCES = {
    os.path.join(ROOT, "2_xgboost_model", "level1_binary_xgb.json"): os.path.join(MODELS, "level1_binary_xgb.json"),
    os.path.join(ROOT, "2_xgboost_model", "level2_multiclass_xgb.json"): os.path.join(MODELS, "level2_multiclass_xgb.json"),
    os.path.join(ROOT, "2_xgboost_model", "data", "meta.json"): os.path.join(MODELS, "meta.json"),
    os.path.join(ROOT, "2_xgboost_model", "data", "attack_classes.json"): os.path.join(MODELS, "attack_classes.json"),
    # hybrid model (Isolation Forest + XGBoost stage 2)
    os.path.join(ROOT, "isolation_forest_and_xgboost_model", "isolation_forest.joblib"): os.path.join(MODELS, "isolation_forest.joblib"),
    os.path.join(ROOT, "isolation_forest_and_xgboost_model", "scaler.joblib"): os.path.join(MODELS, "scaler.joblib"),
    os.path.join(ROOT, "isolation_forest_and_xgboost_model", "model_config.json"): os.path.join(MODELS, "hybrid_config.json"),
    os.path.join(ROOT, "isolation_forest_and_xgboost_model", "xgboost_stage2.json"): os.path.join(MODELS, "xgboost_stage2.json"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(MODELS, exist_ok=True)
    lines = []
    for src, dst in SOURCES.items():
        before = sha256(dst) if os.path.exists(dst) else None
        shutil.copyfile(src, dst)
        after = sha256(dst)
        state = "unchanged" if before == after else ("NEW" if before is None else "UPDATED")
        print(f"{state:<10} {os.path.basename(dst)}")
        lines.append(f"{after}  {os.path.basename(dst)}")
    with open(os.path.join(MODELS, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote models/SHA256SUMS")


if __name__ == "__main__":
    main()
