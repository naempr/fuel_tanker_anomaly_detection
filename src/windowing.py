
import numpy as np

def _encode_feature_series(series, name):
    if name == "geofence_status":
        return (series.astype(str) == "depot").astype(float).values
    return series.astype(float).values

def make_windows(df_trip, window_sec, stride_sec, sample_period_s, feat_cols):
    T = len(df_trip)
    win = int(window_sec // sample_period_s)
    st  = int(stride_sec // sample_period_s)
    if win <= 0 or st <= 0 or T < win:
        return np.empty((0,)), np.empty((0,)), [], feat_cols

    mats = [_encode_feature_series(df_trip[c], c) for c in feat_cols]
    M = np.stack(mats, axis=1)  # [T, F]
    labels = df_trip["label"].astype(str).values

    Xs, ys, starts = [], [], []
    for s in range(0, T - win + 1, st):
        e = s + win
        Xs.append(M[s:e, :])
        wl = labels[s:e]
        if "theft" in wl: y = "theft"
        elif "delivery" in wl: y = "delivery"
        else: y = "normal"
        ys.append(y); starts.append(s)

    X = np.stack(Xs, axis=0) if Xs else np.empty((0, win, len(feat_cols)))
    y = np.array(ys) if ys else np.empty((0,))
    return X, y, starts, feat_cols
