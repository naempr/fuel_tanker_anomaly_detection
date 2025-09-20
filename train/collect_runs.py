
import os, argparse, json, pandas as pd

def parse_report(report_txt):
    rows = {}
    with open(report_txt,"r",encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts)>=5 and parts[0] not in ["accuracy","macro","weighted"]:
                try:
                    rows[f"{parts[0]}_precision"]=float(parts[1])
                    rows[f"{parts[0]}_recall"]=float(parts[2])
                    rows[f"{parts[0]}_f1"]=float(parts[3])
                except:
                    pass
    return rows

def collect_run(run_dir):
    out = {"run_dir": run_dir}
    rep = os.path.join(run_dir,"eval","classification_report.txt")
    if os.path.exists(rep):
        out.update({f"attn_{k}":v for k,v in parse_report(rep).items()})
    rep2 = os.path.join(run_dir,"rf_eval","classification_report.txt")
    if os.path.exists(rep2):
        out.update({f"rf_{k}":v for k,v in parse_report(rep2).items()})
    man = os.path.join(run_dir,"manifest.json")
    if os.path.exists(man):
        with open(man,"r",encoding="utf-8") as f:
            m = json.load(f)
        out["scenario"] = m.get("scenario")
        out["run_tag"] = m.get("run_tag")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", default="artifacts")
    p.add_argument("--out_csv", default="runs/summary_all_runs.csv")
    a = p.parse_args()

    rows = []
    for scenario in sorted(os.listdir(a.artifacts_dir)):
        scen_dir = os.path.join(a.artifacts_dir, scenario)
        if not os.path.isdir(scen_dir): continue
        for run_tag in sorted(os.listdir(scen_dir)):
            d = os.path.join(scen_dir, run_tag)
            if os.path.isdir(d):
                rows.append(collect_run(d))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    print("Wrote", a.out_csv, "rows:", len(df))

if __name__ == "__main__":
    main()
