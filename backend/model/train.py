import os
import sys
import traceback

from train_common import train_model


if __name__ == "__main__":
    try:
        train_model(
            {
                "model_label": "特征增强LSTM",
                "feature_mode": "featured",
                "loss_mode": "mae_day_weighted",
                "model_file": "lstm_pm25_changchun_featured_direct7.pt",
                "scaler_file": "scaler_pm25_featured_direct7.save",
                "high_pollution_eval": False,
            }
        )

        print("TRAIN_PROCESS_EXIT - code: 0", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    except Exception as e:
        print(f"TRAIN_PROCESS_ERROR - {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)