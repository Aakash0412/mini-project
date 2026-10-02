from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from src.transfer.neorl_source_policy import NeoRLSourcePolicy,NeoRLSourcePolicyConfig
from src.transfer.transfer_adapter import TransferAdapter,TransferAdapterConfig

DEVICE=torch.device("cpu")

def load_models():
    source=NeoRLSourcePolicy(NeoRLSourcePolicyConfig()).to(DEVICE)
    p=torch.load(PROJECT_ROOT/"models/transfer/neorl_source_policy/best_model.pth",map_location=DEVICE,weights_only=False)
    source.load_state_dict(p.get("model_state_dict",p)); source.eval()
    adapter=TransferAdapter(TransferAdapterConfig()).to(DEVICE)
    p=torch.load(PROJECT_ROOT/"models/transfer/adapter/best_adapter.pth",map_location=DEVICE,weights_only=False)
    adapter.load_state_dict(p.get("model_state_dict",p)); adapter.eval()
    return source,adapter

def load_data():
    d=PROJECT_ROOT/"data/processed/state"
    states=np.load(d/"state.npy").astype(np.float32)
    rows=np.load(d/"state_row_indices.npy").astype(np.int64)
    products=pd.read_parquet(PROJECT_ROOT/"data/processed/products_clean.parquet")
    months=pd.to_numeric(products.iloc[rows]["MonthNum"],errors="coerce").to_numpy()
    return states,(
        np.isfinite(months)&(months<=6),
        np.isfinite(months)&(months>=7)&(months<=8),
        np.isfinite(months)&(months>=9)&(months<=10)
    )

def stats(name,x):
    x=np.asarray(x).reshape(-1)
    return {"mean":float(x.mean()),"std":float(x.std()),"min":float(x.min()),
            "p01":float(np.percentile(x,1)),"p05":float(np.percentile(x,5)),
            "p50":float(np.percentile(x,50)),"p95":float(np.percentile(x,95)),
            "p99":float(np.percentile(x,99)),"max":float(x.max())}

@torch.no_grad()
def audit(name,states,mask,adapter,source):
    idx=np.where(mask)[0]; ps=[]; means=[]; logs=[]; det=[]
    for s in range(0,len(idx),512):
        x=torch.from_numpy(states[idx[s:s+512]]).to(DEVICE)
        z=adapter(x); out=source.forward(z)
        if not isinstance(out,tuple): raise RuntimeError("source forward must return (mean, log_std)")
        m,ls=out[:2]
        ps.append(z.cpu().numpy()); means.append(m.cpu().numpy()); logs.append(ls.cpu().numpy())
        det.append(source.deterministic_action(z).cpu().numpy())
    ps=np.concatenate(ps); m=np.concatenate(means); ls=np.concatenate(logs); da=np.concatenate(det)
    factorA=m[:,1]*5.0
    discountA=(1-factorA)*100
    factorB=m[:,1]
    discountB=(1-factorB)*100
    print("\n"+"="*72); print(name.upper()); print("="*72)
    print("Samples:",len(idx))
    for j in range(4): print(f"Pseudo-state {j}: {stats('',ps[:,j])}")
    for j in range(2): print(f"Source mean action {j}: {stats('',m[:,j])}")
    for j in range(2): print(f"Source log-std {j}: {stats('',ls[:,j])}")
    for j in range(2): print(f"Deterministic action {j}: {stats('',da[:,j])}")
    print("Discount interpretation A (mean*5):",stats('',discountA))
    print("Discount interpretation B (mean direct):",stats('',discountB))
    pairs,c=np.unique(np.round(da,6),axis=0,return_counts=True)
    order=np.argsort(c)[::-1]
    print("Top deterministic action pairs:")
    for k in order[:15]: print(" ",pairs[k].tolist(),int(c[k]),f"{100*c[k]/len(idx):.3f}%")
    return {"samples":int(len(idx)),"pseudo_mean":ps.mean(0).tolist(),"pseudo_std":ps.std(0).tolist(),
            "source_mean_stats":[stats("",m[:,j]) for j in range(2)],
            "source_log_std_stats":[stats("",ls[:,j]) for j in range(2)],
            "deterministic_stats":[stats("",da[:,j]) for j in range(2)],
            "discount_A":stats("",discountA),"discount_B":stats("",discountB)}

def main():
    print("="*72); print("NEORL SOURCE-POLICY ACTION-SEMANTICS AUDIT"); print("="*72)
    print("Diagnostic only; no model is modified.")
    source,adapter=load_models(); states,masks=load_data()
    print("Aligned states:",states.shape)
    names=["train","validation","test"]; results={}
    for n,mask in zip(names,masks): results[n]=audit(n,states,mask,adapter,source)
    out=PROJECT_ROOT/"reports/transfer/neorl_source_policy_action_audit.json"; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("\nAudit complete:",out)

if __name__=="__main__": main()
