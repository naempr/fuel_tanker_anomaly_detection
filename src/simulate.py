import numpy as np
import pandas as pd

def simulate_trip(cfg, rng):
    """
    Physics-aware simulator for tanker trips with clean labeling.
    - Events never overlap (delivery/theft).
    - Delivery only in final depot segment, theft only in road.
    - Theft requires BOTH (low speed) AND (min stop duration).
    - Events are clipped to the current geofence segment.
    - Exact fuel balance; gauge = true fuel + sloshing AR(1) + white noise.
    """
    sim = cfg["simulation"]

    # ---------- Timing ----------
    dt = float(sim["sample_period_s"])
    minutes = int(rng.integers(sim["min_trip_min"], sim["max_trip_min"] + 1))
    T = int(minutes * 60 // dt)

    # ---------- Geofence (depot -> road -> depot) ----------
    depot_len = max(1, int(T * sim["depot_zone_ratio"]))
    geofence = np.array(["depot"] * depot_len + ["road"] * (T - 2 * depot_len) + ["depot"] * depot_len)
    # helper: پیدا کردن انتهای قطعه‌ی ژئوفنس از اندیس t
    def seg_end_ix(t):
        g = geofence[t]
        i = t
        while i < T and geofence[i] == g:
            i += 1
        return i  # اندیس اولین نمونه بعد از قطعه‌ی فعلی

    # ---------- Speed + random stops ----------
    speed = np.clip(rng.normal(sim["base_speed_kmh"], sim["speed_std_kmh"], size=T), 0.0, None)

    stop_mask = np.zeros(T, dtype=bool)
    i = 0
    while i < T:
        p_stop_here = 0.05 if geofence[i] == "road" else 0.08
        if rng.random() < p_stop_here:
            dur = int(rng.integers(sim["min_stop_s"], sim["max_stop_s"] + 1) // dt)
            stop_mask[i:i + dur] = True
            i += dur
        else:
            i += 1
    speed[stop_mask] = 0.0

    # cumulative stop duration (s)
    stop_duration = np.zeros(T, dtype=float)
    acc_stop = 0.0
    for t in range(T):
        if speed[t] < 0.5:
            acc_stop += dt
        else:
            acc_stop = 0.0
        stop_duration[t] = acc_stop

    # ---------- Init series ----------
    label = np.array(["normal"] * T, dtype=object)
    flow  = np.zeros(T, dtype=float)
    occupied = np.zeros(T, dtype=bool)  # ### CHANGED: برای جلوگیری از هم‌پوشانی

    # ---------- Event rates (per minute -> per step) ----------
    lam_del = float(sim.get("lambda_delivery_per_min", 0.8)) * float(sim.get("delivery_prob", 1.0))
    lam_th  = float(sim.get("lambda_theft_per_min",    0.3)) * float(sim.get("theft_prob",    1.0))
    p_del = 1.0 - np.exp(-lam_del * (dt / 60.0))
    p_th  = 1.0 - np.exp(-lam_th  * (dt / 60.0))

    # ---------- Flow params ----------
    flow_delivery_mu  = float(sim["flow_lpm_delivery"])
    flow_delivery_std = float(sim.get("flow_delivery_std", 20.0))

    theft_flow_cfg = sim["flow_lpm_theft"]
    theft_max_speed = float(sim.get("theft_max_speed_kmh", 3.0))    # ### CHANGED: واقع‌گرایانه
    theft_min_stop  = float(sim.get("theft_min_stop_s", 10.0))      # ### CHANGED: واقع‌گرایانه

    # indices depot head/tail
    depot_head_end = depot_len
    depot_tail_start = T - depot_len

    # ---------- DELIVERY: only in final depot segment ----------
    t = depot_tail_start
    while t < T:
        if geofence[t] == "depot" and not occupied[t] and stop_duration[t] >= sim["min_stop_s"]:
            if rng.random() < p_del:
                # مدت رویداد + کلیپ به انتهای قطعه‌ی depot
                dur = int(rng.integers(sim["min_stop_s"], sim["max_stop_s"] + 1) // dt)
                seg_end = seg_end_ix(t)
                te = min(T, t + dur, seg_end)
                if te > t:
                    deliv_lpm = max(0.0, rng.normal(flow_delivery_mu, flow_delivery_std))
                    # رزرو و اعمال
                    flow[t:te] += deliv_lpm
                    label[t:te] = "delivery"
                    occupied[t:te] = True
                    t = te
                    continue
        t += 1

    # ---------- THEFT: road + low speed AND min stop, no overlap ----------
    t = depot_head_end  # از پایان depot ابتدایی به بعد (یعنی از road شروع می‌شود)
    while t < depot_tail_start:  # تا قبل depot پایانی
        if geofence[t] == "road" and not occupied[t]:
            cond_speed = (speed[t] <= theft_max_speed)
            cond_stop  = (stop_duration[t] >= theft_min_stop)
            if cond_speed and cond_stop and (rng.random() < p_th):
                dur = int(rng.integers(sim["min_stop_s"], sim["max_stop_s"] + 1) // dt)
                seg_end = seg_end_ix(t)  # اینجا قطعاً road است
                te = min(T, t + dur, seg_end)
                if te > t:
                    if isinstance(theft_flow_cfg, (list, tuple, np.ndarray)) and len(theft_flow_cfg) == 2:
                        theft_lpm = rng.uniform(float(theft_flow_cfg[0]), float(theft_flow_cfg[1]))
                    else:
                        theft_lpm = float(theft_flow_cfg)
                    theft_lpm = max(0.0, theft_lpm)
                    # رزرو و اعمال
                    flow[t:te] += theft_lpm
                    label[t:te] = "theft"
                    occupied[t:te] = True
                    t = te
                    continue
        t += 1

    # ---------- Small leak (does NOT change label) ----------
    leak_prob = float(sim["leak_prob"])
    if rng.random() < leak_prob:
        leak_low, leak_high = sim["small_leak_lpm"]
        leak_lpm = rng.uniform(float(leak_low), float(leak_high))
        flow += leak_lpm

    # ---------- Fuel dynamics (exact balance, clipped at 0) ----------
    fuel = np.zeros(T, dtype=float)
    fs_min, fs_max = sim["fuel_start_ratio"]
    fuel[0] = rng.uniform(float(fs_min), float(fs_max)) * float(sim["fuel_capacity_l"])

    for t in range(1, T):
        max_out_lpm = fuel[t-1] * (60.0 / dt)  # حداکثر دبی که باعث منفی شدن نشود
        eff_flow = min(flow[t-1], max_out_lpm) if max_out_lpm > 0 else 0.0
        fuel[t] = max(0.0, fuel[t-1] - eff_flow * (dt / 60.0))

    # ---------- Gauge = true + sloshing AR(1) + white ----------
    v_ms = speed * (1000.0 / 3600.0)
    a_ms2 = np.zeros(T, dtype=float)
    a_ms2[1:] = (v_ms[1:] - v_ms[:-1]) / dt

    rho   = float(sim.get("sloshing_rho", 0.90))
    sigma = float(sim.get("sloshing_sigma_l", 1.5))
    kacc  = float(sim.get("sloshing_acc_gain_l", 2.0))

    slosh = np.zeros(T, dtype=float)
    for t in range(1, T):
        slosh[t] = rho * slosh[t-1] + sigma * rng.normal() + kacc * np.tanh(a_ms2[t])

    gauge_noise = float(sim["fuel_gauge_noise_l"])
    fuel_gauge  = fuel + slosh + rng.normal(0.0, gauge_noise, size=T)

    dFuel = np.zeros(T, dtype=float)
    dFuel[1:] = (fuel_gauge[1:] - fuel_gauge[:-1]) / (dt / 60.0)

    df = pd.DataFrame({
        "time_s": np.arange(T) * dt,
        "trip_id": 0,
        "geofence_status": geofence,          # "road"/"depot" (downstream اگر لازم شد 0/1 کن)
        "speed_kmh": speed,
        "flow_lpm": flow,                     # L/min (true outflow)
        "fuel_l": fuel_gauge,                 # gauge reading (true + slosh + noise)
        "fuel_change_rate_lpm": dFuel,        # d(fuel_gauge)/dt in L/min
        "stop_duration_s": stop_duration,
        "label": label
    })
    return df
