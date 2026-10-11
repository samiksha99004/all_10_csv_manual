LIVE IDS APP - HOW TO RUN
=========================
Nothing in this folder has been run yet. Follow these steps in order, on the machine that
will do the live detection (a Kali VM is what this project has used and tested before).

WHAT IT DOES
------------
Captures real network traffic, converts it to flow features with the Java CICFlowMeter-v3
(the real converter CICIDS2018 was built with), and every 5 seconds feeds the new flows to
the trained model (new_isolationforest_xgboost_hybrid) and prints what the network looks
like: Benign (normal) or one of 14 attack types (DoS, DDoS, Bot, SSH-Bruteforce, etc.).

Only the 63 features the model was trained on are used, cast to the same data type used in
training (float32). A flow is scored only if every one of those 63 values is present and a
real number - a flow missing a value is dropped and counted, never filled in with a fake
number or left blank. This is enforced in 3_predict.py, not just claimed.

STEP 1 - JAVA CONVERTER (one-time)
-----------------------------------
If this machine does not already have the Java CICFlowMeter-v3 built (check:
ls ~/live_ids_app/CICFlowMeter-4.0), copy it from wherever it was built before in this
project (see documents/CLAUDE.md / ssh_plan.txt for where that was done earlier), e.g.:
    mkdir -p ~/live_ids_app
    scp -r <other_machine>:/path/to/CICFlowMeter-4.0 ~/live_ids_app/

If the converter fails with "libpcap.so: cannot open shared object file", this project hit
that before. Fix (adjust the .so path if different):
    mkdir -p ~/live_ids_app/libs
    ln -sf /usr/lib/x86_64-linux-gnu/libpcap.so.0.8 ~/live_ids_app/libs/libpcap.so
    export LD_LIBRARY_PATH=$HOME/live_ids_app/libs

STEP 2 - PYTHON PACKAGES
--------------------------
    python3 -c "import numpy, xgboost, joblib, sklearn" || \
      pip install --break-system-packages numpy xgboost joblib scikit-learn

STEP 3 - COPY THE MODEL FILES
--------------------------------
Copy these 4 files from new_isolationforest_xgboost_hybrid/ (on the machine that has the
project) into this folder's models/ directory:
    isolation_forest.joblib
    scaler.joblib
    xgboost_model.json
    if_config.json
(Use the same shared-folder / scp method already used elsewhere in this project.)

STEP 4 - CHECK THE NETWORK INTERFACE
---------------------------------------
    ip -br a
Note the interface name with this machine's real IP (e.g. eth0).

STEP 5 - RUN THE THREE PARTS (three terminals, all on this machine)
-----------------------------------------------------------------------
Terminal 1 - start the rotating capture (writes a new pcap every 5 seconds):
    sudo bash 1_capture.sh eth0

Terminal 2 - start the converter loop (turns each finished pcap into a flow CSV):
    export LD_LIBRARY_PATH=$HOME/live_ids_app/libs   (only if step 1's fix was needed)
    bash 2_convert_loop.sh

Terminal 3 - start the predictor (reads new flow CSVs, scores them, prints every 5 seconds):
    python3 3_predict.py

Leave all three running. Every 5 seconds, terminal 3 prints a block like:
    [14:03:21] window: 9 flows captured, 9 scored, 0 dropped
      NETWORK STATUS: Benign
        Benign                          9  100.0%
      most active source IP this window: 192.168.1.10
      cumulative: 9 flows scored, 0 dropped, since start

If no traffic happened in a 5-second window, it prints "no new flows this window -
NETWORK STATUS: (no data)" - it will never guess a status with nothing to score.

CHECKING RESULTS LATER
-----------------------
Every result is saved, not just printed. While terminal 3 runs (and after you stop it):
    results/flow_log.csv     one row per scored flow - time, source/destination, predicted
                              class, status (normal = Benign, abnormal = any attack type),
                              and remote_host (a reverse-DNS name for abnormal flows only,
                              e.g. an IP that is really github.com - so you can tell a real
                              attack from ordinary traffic to a known service)
    results/window_log.csv   one row per 5-second window - counts and that window's verdict
Both files are appended to and flushed after every write, so you can open flow_log.csv at
any time (even while the app is still running) to see exactly which flows were called
normal and which were called abnormal, and what they were named.
The reverse-DNS lookup is best-effort: some IPs (especially CDN/anycast addresses) have no
reverse-DNS record, so remote_host is left blank for those rather than guessing a name.

STOPPING
--------
Ctrl+C in each terminal, in this order: terminal 1 (capture) first, then terminal 2
(converter), then terminal 3 (predictor). Terminal 3 prints a final cumulative summary.

NOTES
-----
- This app only reads the 63 columns in app_common.py's JAVA_COLUMN mapping from the
  converter's CSV. It does not write a separate trimmed CSV of its own - the converter's
  CSV in ~/live_ids_app/csv/ already is "the live network features in a CSV file"; this
  script just selects the 63 trained columns from it when scoring.
- Known open item: the model's own offline test shows the Infilteration class is still
  weak (49.5% recall - see new_isolationforest_xgboost_hybrid's training_report.pdf), and
  live captures with very short/few-packet flows have been missed by every model trained
  in this project so far. This app will show whatever the model actually outputs, including
  a wrong or missed call - it does not hide or correct that.
