"""Shared settings for live_ids_app. Import this first (WMI patch for Windows; harmless on Linux)."""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")          # copy the 4 model files here before running (see README_RUN.txt)
RESULTS_DIR = os.path.join(HERE, "results")    # 3_predict.py writes flow_log.csv and window_log.csv here
PCAP_DIR = os.path.join(HERE, "pcap")
CSV_DIR = os.path.join(HERE, "csv")
if sys.platform.startswith("win"):
    # This machine's already-built Windows CICFlowMeter-4.0 distribution (extracted 2026-10-11).
    CONV_DIR = r"F:\CICFlowMeter-master\build\distributions\CICFlowMeter-4.0"
else:
    CONV_DIR = os.path.expanduser("~/live_ids_app/CICFlowMeter-4.0")  # one-time copy on Kali

WINDOW_SECONDS = 5          # how often a prediction is printed
KEEP_FILES = 40             # pcap/csv retention: keep the newest this many
MAX_FILES = 60              # ... delete the oldest once more than this many exist

# Training feature name -> Java CICFlowMeter-v3 CSV header name.
# Restricted to the 63 features new_isolationforest_xgboost_hybrid was trained on
# (new_isolationforest_xgboost_hybrid/if_config.json "features_used"). Verified against a real
# Java CICFlowMeter-v3 output file earlier in this project (exact text match, except this one
# known quirk: the live tool spells "CWR Flag Count", training data spells it "CWE Flag Count").
JAVA_COLUMN = {
    "Dst Port": "Dst Port", "Protocol": "Protocol", "Flow Duration": "Flow Duration",
    "Tot Fwd Pkts": "Total Fwd Packet", "Tot Bwd Pkts": "Total Bwd packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packet", "TotLen Bwd Pkts": "Total Length of Bwd Packet",
    "Fwd Pkt Len Max": "Fwd Packet Length Max", "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean", "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max", "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean", "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s", "Flow Pkts/s": "Flow Packets/s",
    "Flow IAT Mean": "Flow IAT Mean", "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max", "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Tot": "Fwd IAT Total", "Fwd IAT Mean": "Fwd IAT Mean", "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max", "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Tot": "Bwd IAT Total", "Bwd IAT Mean": "Bwd IAT Mean", "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max", "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags", "Fwd URG Flags": "Fwd URG Flags",
    "Fwd Header Len": "Fwd Header Length", "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s", "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Packet Length Min", "Pkt Len Max": "Packet Length Max",
    "Pkt Len Mean": "Packet Length Mean", "Pkt Len Std": "Packet Length Std",
    "FIN Flag Cnt": "FIN Flag Count", "SYN Flag Cnt": "SYN Flag Count", "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count", "ACK Flag Cnt": "ACK Flag Count", "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Count": "CWR Flag Count", "ECE Flag Cnt": "ECE Flag Count",
    "Down/Up Ratio": "Down/Up Ratio", "Pkt Size Avg": "Average Packet Size",
    "Init Fwd Win Byts": "FWD Init Win Bytes", "Init Bwd Win Byts": "Bwd Init Win Bytes",
    "Fwd Act Data Pkts": "Fwd Act Data Pkts", "Fwd Seg Size Min": "Fwd Seg Size Min",
    "Active Mean": "Active Mean", "Active Std": "Active Std", "Active Max": "Active Max", "Active Min": "Active Min",
    "Idle Mean": "Idle Mean", "Idle Std": "Idle Std", "Idle Max": "Idle Max", "Idle Min": "Idle Min",
}
