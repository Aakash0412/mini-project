#!/usr/bin/env python3
"""Evidence-first paper-comparable evaluation for Dynamic Pricing for E-commerce.

Read-only with respect to project inputs/checkpoints. Writes only to a new output
folder and refuses to overwrite it. No model training is performed.

Important: local profit is a gross-margin proxy; the current environment is one-step.
This script will not label a sum of one-step rewards as sequential cumulative reward.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

PAPER = {
    "title": "A Multimodal Deep Reinforcement Learning Framework for Dynamic Pricing Optimisation in E-Commerce",
    "doi": "10.1109/ACCESS.2026.3680140",
    "test_samples": 6810,
    "profit_margin_growth_reference": ">16.7%",
    "sales_volume_change_reference": "-0.3% to -2.0%",
    "ppo_top1_profit_count": 3563, "ppo_top1_profit_pct": 52.32,
    "ppo_top3_sales_count": 5062, "ppo_top3_sales_pct": 74.33,
}


def args_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-csv', type=Path, default=Path('models/ppo_transfer/transfer_supported_8action_policy_evaluation.csv'))
    p.add_argument('--dataset-parquet', type=Path, default=Path('data/processed/products_clean.parquet'))
    p.add_argument('--product-metrics-csv', type=Path, default=Path('reports/algorithm_comparison/paper_aligned_metrics/per_product_growth.csv'))
    p.add_argument('--robustness-csv', type=Path, default=Path('reports/robustness/multiseed_aggregate.csv'))
    p.add_argument('--output-dir', type=Path, default=Path('reports/base_paper_comparison_v3'))
    p.add_argument('--bootstrap-replicates', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def require(df, cols, path):
    missing = [c for c in cols if c not in df.columns]
    if missing: raise ValueError(f'{path} missing required columns: {missing}')


def read_inputs(a):
    for p in [a.evaluation_csv, a.dataset_parquet]:
        if not p.exists(): raise FileNotFoundError(f'Required input not found: {p}')
    ev = pd.read_csv(a.evaluation_csv)
    require(ev, ['split','source_row','ppo_reward','ppo_sales','ppo_margin','oracle_reward','oracle_sales','oracle_margin','baseline_sales','baseline_margin','ppo_relative_delta_quantity','ppo_relative_delta_margin'], a.evaluation_csv)
    test = ev.loc[ev['split'].astype(str).str.lower().eq('test')].copy()
    if test.empty: raise ValueError('No split=test rows found in evaluation CSV.')
    test['source_row'] = pd.to_numeric(test['source_row'], errors='raise').astype(int)
    if test['source_row'].duplicated().any(): raise ValueError('Duplicate source_row in test split; refusing ambiguous alignment.')
    data = pd.read_parquet(a.dataset_parquet)
    require(data, ['ASIN','MonthNum','Estimated Monthly Sales Growth Rate','Discount'], a.dataset_parquet)
    if test.source_row.min() < 0 or test.source_row.max() >= len(data):
        raise ValueError('source_row is outside the cleaned dataset row-position range; cannot safely map to ASIN/month.')
    mapped = data.iloc[test.source_row.to_numpy()].reset_index(drop=True)
    test = test.reset_index(drop=True)
    test['ASIN'] = mapped['ASIN'].astype(str)
    test['MonthNum'] = pd.to_numeric(mapped['MonthNum'], errors='coerce')
    months = sorted(test['MonthNum'].dropna().astype(int).unique().tolist())
    if months != [9, 10]:
        raise ValueError(f'Test source_row mapping produced months {months}, not expected [9, 10]. Stop to prevent invalid product alignment.')
    return test, data, months


def num(s): return pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)


def growth_arrays(outcome, baseline):
    o, b = num(outcome), num(baseline)
    valid = np.isfinite(o) & np.isfinite(b) & (b > 0)
    row_growth = np.full(len(o), np.nan)
    row_growth[valid] = (o[valid] / b[valid] - 1) * 100
    ratio = (o[valid].sum() / b[valid].sum() - 1) * 100 if valid.any() and b[valid].sum() > 0 else np.nan
    mean = float(np.mean(row_growth[valid])) if valid.any() else np.nan
    return row_growth, float(ratio), mean, int(valid.sum()), valid


def metric_rows(test):
    sr, sagg, smean, sn, sv = growth_arrays(test.ppo_sales, test.baseline_sales)
    mr, magg, mmean, mn, mv = growth_arrays(test.ppo_margin, test.baseline_margin)
    reward = num(test.ppo_reward); oracle = num(test.oracle_reward)
    gap = oracle - reward
    rows = []
    def add(metric, value, unit, n, status='local one-step simulation'):
        if isinstance(value, (str, np.str_)):
            stored = str(value)
        else:
            stored = float(value) if np.isfinite(value) else np.nan
        rows.append({'metric':metric,'value':stored,'unit':unit,'valid_n':int(n),'status':status})
    add('test_observations',len(test),'rows',len(test))
    add('unique_test_ASINs',test.ASIN.nunique(),'ASINs',len(test))
    add('test_months',';'.join(map(str,sorted(test.MonthNum.dropna().astype(int).unique()))),'months',len(test))
    add('mean_reward_per_observation',np.nanmean(reward),'reward',np.isfinite(reward).sum())
    add('sum_one_step_rewards',np.nansum(reward),'reward; NOT sequential cumulative reward',np.isfinite(reward).sum(),'descriptive sum only; not paper-equivalent cumulative reward')
    add('mean_per_row_sales_growth',smean,'% per row',sn)
    add('aggregate_sales_growth_ratio_of_totals',sagg,'% aggregate',sn)
    add('mean_per_row_margin_proxy_growth',mmean,'% per row',mn,'gross-margin proxy; not accounting profit')
    add('aggregate_margin_proxy_growth_ratio_of_totals',magg,'% aggregate',mn,'gross-margin proxy; not accounting profit')
    add('oracle_mean_reward',np.nanmean(oracle),'reward',np.isfinite(oracle).sum(),'simulated oracle reference')
    add('PPO_oracle_mean_reward_ratio',np.nanmean(reward)/np.nanmean(oracle)*100 if np.nanmean(oracle)!=0 else np.nan,'% of oracle reward',np.isfinite(reward).sum(),'simulator-relative; not real-world performance')
    add('mean_oracle_reward_gap',np.nanmean(gap),'reward',np.isfinite(gap).sum(),'oracle reward minus PPO reward')
    if 'oracle_action_match' in test.columns:
        match = test.oracle_action_match
        add('oracle_action_match_pct',pd.to_numeric(match,errors='coerce').mean()*100,'% of rows',match.notna().sum())
    elif {'ppo_action','oracle_action'}.issubset(test.columns):
        add('oracle_action_match_pct',(test.ppo_action==test.oracle_action).mean()*100,'% of rows',len(test))
    return pd.DataFrame(rows), {'sales_row_growth':sr,'margin_row_growth':mr,'sales_valid':sv,'margin_valid':mv}


def bootstrap_asin(test, arrays, reps, seed):
    """Cluster bootstrap over ASINs; all rows for a sampled ASIN travel together."""
    rng = np.random.default_rng(seed)
    work = test.copy()
    work['_reward'] = num(work.ppo_reward)
    work['_oracle_gap'] = num(work.oracle_reward) - num(work.ppo_reward)
    work['_sales_growth'] = arrays['sales_row_growth']
    work['_margin_growth'] = arrays['margin_row_growth']
    work['_sales_out'] = num(work.ppo_sales); work['_sales_base'] = num(work.baseline_sales)
    work['_margin_out'] = num(work.ppo_margin); work['_margin_base'] = num(work.baseline_margin)
    work['_sales_ok'] = arrays['sales_valid']; work['_margin_ok'] = arrays['margin_valid']
    clusters = [g for _,g in work.groupby('ASIN', sort=False)]
    if len(clusters) < 2: return pd.DataFrame(), len(clusters)
    # Aggregate sufficient statistics per ASIN, then resample ASIN clusters.
    agg=[]
    for g in clusters:
        def sm(col): return np.nansum(g[col].to_numpy(dtype=float))
        def cnt(col): return int(g[col].sum())
        sales_ok = g['_sales_ok'].to_numpy(dtype=bool)
        margin_ok = g['_margin_ok'].to_numpy(dtype=bool)
        agg.append([sm('_reward'),int(np.isfinite(g['_reward']).sum()),sm('_oracle_gap'),int(np.isfinite(g['_oracle_gap']).sum()),
                    np.nansum(g['_sales_out'].to_numpy(dtype=float)[sales_ok]) if sales_ok.any() else 0,
                    np.nansum(g['_sales_base'].to_numpy(dtype=float)[sales_ok]) if sales_ok.any() else 0,cnt('_sales_ok'),
                    np.nansum(g['_margin_out'].to_numpy(dtype=float)[margin_ok]) if margin_ok.any() else 0,
                    np.nansum(g['_margin_base'].to_numpy(dtype=float)[margin_ok]) if margin_ok.any() else 0,cnt('_margin_ok'),
                    sm('_sales_growth'),cnt('_sales_ok'),sm('_margin_growth'),cnt('_margin_ok')])
    a=np.asarray(agg,dtype=float); n=len(a); samples={'mean_reward':[],'mean_oracle_gap':[],'aggregate_sales_growth_pct':[],'aggregate_margin_proxy_growth_pct':[],'mean_row_sales_growth_pct':[],'mean_row_margin_proxy_growth_pct':[]}
    for _ in range(reps):
        ix=rng.integers(0,n,size=n); b=a[ix].sum(axis=0)
        samples['mean_reward'].append(b[0]/b[1] if b[1] else np.nan)
        samples['mean_oracle_gap'].append(b[2]/b[3] if b[3] else np.nan)
        samples['aggregate_sales_growth_pct'].append((b[4]/b[5]-1)*100 if b[6] and b[5]>0 else np.nan)
        samples['aggregate_margin_proxy_growth_pct'].append((b[7]/b[8]-1)*100 if b[9] and b[8]>0 else np.nan)
        samples['mean_row_sales_growth_pct'].append(b[10]/b[11] if b[11] else np.nan)
        samples['mean_row_margin_proxy_growth_pct'].append(b[12]/b[13] if b[13] else np.nan)
    rows=[]
    for k,vals in samples.items():
        vals=np.asarray(vals,dtype=float); vals=vals[np.isfinite(vals)]
        rows.append({'metric':k,'cluster_unit':'ASIN','clusters':n,'bootstrap_replicates_requested':reps,'valid_replicates':len(vals),'ci95_low':np.percentile(vals,2.5) if len(vals) else np.nan,'ci95_high':np.percentile(vals,97.5) if len(vals) else np.nan,'bootstrap_mean':np.mean(vals) if len(vals) else np.nan})
    return pd.DataFrame(rows),n


def product_rankings(path):
    if not path.exists(): return None
    df=pd.read_csv(path)
    if {'source_row','algorithm','sales_growth_pct','margin_proxy_growth_pct'}.issubset(df.columns): pass
    elif {'source_row','method','sales_growth_pct','margin_proxy_growth_pct'}.issubset(df.columns): df=df.rename(columns={'method':'algorithm'})
    else: raise ValueError(f'{path} schema not supported. Need source_row, algorithm/method, sales_growth_pct, margin_proxy_growth_pct.')
    df=df[['source_row','algorithm','sales_growth_pct','margin_proxy_growth_pct']].copy()
    df['source_row']=pd.to_numeric(df.source_row,errors='raise').astype(int); df['algorithm']=df.algorithm.astype(str).str.upper()
    for c in ['sales_growth_pct','margin_proxy_growth_pct']: df[c]=pd.to_numeric(df[c],errors='coerce')
    rows=[]
    for metric in ['margin_proxy_growth_pct','sales_growth_pct']:
        v=df.dropna(subset=['source_row','algorithm',metric]).copy()
        if v.duplicated(['source_row','algorithm']).any(): raise ValueError(f'Duplicate source_row/algorithm rows in {path} for {metric}')
        v['rank']=v.groupby('source_row')[metric].rank(method='min',ascending=False)
        for alg,g in v.groupby('algorithm'):
            n=g.source_row.nunique()
            rows.append({'metric':metric,'algorithm':alg,'n_products_ranked':n,'top1_count':int((g['rank']==1).sum()),'top1_pct':float((g['rank']==1).mean()*100),'top3_count':int((g['rank']<=3).sum()),'top3_pct':float((g['rank']<=3).mean()*100),'ranking_scope':';'.join(sorted(v.algorithm.unique())),'paper_top1_profit_reference_pct':PAPER['ppo_top1_profit_pct'] if metric=='margin_proxy_growth_pct' and alg=='PPO' else np.nan,'paper_top3_sales_reference_pct':PAPER['ppo_top3_sales_pct'] if metric=='sales_growth_pct' and alg=='PPO' else np.nan})
    return pd.DataFrame(rows)


def dataset_audit(data):
    group=data.groupby(['ASIN','MonthNum'],dropna=False)
    sizes=group.size(); dup=sizes[sizes>1]
    target='Estimated Monthly Sales Growth Rate'; discount='Discount'
    variation=group[[target,discount]].nunique(dropna=False) if target in data and discount in data else pd.DataFrame()
    rows=[('dataset_rows',len(data)),('unique_ASINs',data.ASIN.nunique()),('unique_ASIN_month_pairs',data[['ASIN','MonthNum']].drop_duplicates().shape[0]),('duplicate_ASIN_month_groups',len(dup)),('rows_beyond_first_in_duplicate_groups',int((dup-1).sum()) if len(dup) else 0),('ASIN_month_groups_with_multiple_discounts',int((variation[discount]>1).sum()) if len(variation) else 0),('ASIN_month_groups_with_varying_growth_target',int((variation[target]>1).sum()) if len(variation) else 0),('ASINs_seen_multiple_months',int((data.groupby('ASIN').MonthNum.nunique()>1).sum())),('ASINs_seen_one_month',int((data.groupby('ASIN').MonthNum.nunique()==1).sum()))]
    return pd.DataFrame(rows,columns=['audit_item','value'])


def md_table(df):
    if df is None or df.empty:
        return "(no rows)"
    cols = [str(c) for c in df.columns]
    def cell(v):
        if pd.isna(v): return ""
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)

def main():
    a=args_parse()
    if a.bootstrap_replicates < 100: raise ValueError('--bootstrap-replicates must be at least 100 for a useful interval.')
    # Validate/read all inputs before creating output directory, avoiding empty dirs on failure.
    test,data,months=read_inputs(a)
    summary,arrays=metric_rows(test)
    boot,nclusters=bootstrap_asin(test,arrays,a.bootstrap_replicates,a.seed)
    ranking=product_rankings(a.product_metrics_csv)
    audit=dataset_audit(data)
    robustness=None
    if a.robustness_csv.exists():
        r=pd.read_csv(a.robustness_csv)
        required={'noise_fraction','independent_seeds','evaluated_products_per_seed','mean_reward_mean','reward_retention_percent_mean','reward_change_percent_mean'}
        require(r,required,a.robustness_csv)
        keep=[c for c in ['noise_fraction','independent_seeds','evaluated_products_per_seed','mean_reward_mean','mean_reward_ci95_low','mean_reward_ci95_high','reward_retention_percent_mean','reward_change_percent_mean','reward_change_percent_ci95_low','reward_change_percent_ci95_high'] if c in r.columns]
        robustness=r[keep].copy()
    comp=[]
    local={r.metric:r.value for r in summary.itertuples()}
    def add(paper_metric,paper_ref,local_metric,comparability,reason):
        comp.append({'paper_metric':paper_metric,'paper_reference':paper_ref,'local_metric':local_metric,'local_value':local.get(local_metric,np.nan),'comparability':comparability,'interpretation':reason})
    add('Profit-margin growth',PAPER['profit_margin_growth_reference'],'aggregate_margin_proxy_growth_ratio_of_totals','Directional only','Local result is gross-margin proxy, not verified profit; current environment is one-step.')
    add('Sales-volume change',PAPER['sales_volume_change_reference'],'aggregate_sales_growth_ratio_of_totals','Directional only','Local value is simulated growth relative to stored baseline; paper reports sales-volume reduction under its own sequential protocol.')
    add('PPO Top-1 profit-growth frequency',f"{PAPER['ppo_top1_profit_pct']:.2f}% ({PAPER['ppo_top1_profit_count']}/{PAPER['test_samples']})",'PPO Top-1 margin-proxy-growth frequency','Restricted comparison set','Local ranking uses only algorithms in supplied product-metrics CSV; local outcome is a margin proxy.')
    add('PPO Top-3 sales-growth frequency',f"{PAPER['ppo_top3_sales_pct']:.2f}% ({PAPER['ppo_top3_sales_count']}/{PAPER['test_samples']})",'PPO Top-3 sales-growth frequency','Restricted comparison set','Local ranking uses only algorithms in supplied product-metrics CSV.')
    add('Cumulative discounted reward','Sequential discounted reward R(theta)=sum gamma^t r_t','sum_one_step_rewards','Not comparable','Current local environment terminates after one action; sum across products is not a temporal cumulative reward.')
    add('Prediction-noise robustness','Paper protocol requires exact perturbation definition and evaluation alignment','robustness CSV','Partly aligned','Noise-level robustness is available for a fixed policy; this is not robustness across independently trained policies.')
    comparison=pd.DataFrame(comp)
    if ranking is not None:
        for metric,criterion in [('margin_proxy_growth_pct','PPO Top-1 margin-proxy-growth frequency'),('sales_growth_pct','PPO Top-3 sales-growth frequency')]:
            r=ranking[(ranking.metric==metric)&(ranking.algorithm=='PPO')]
            if len(r): comparison.loc[comparison.paper_metric.eq(criterion),'local_value']=float(r.iloc[0].top1_pct if metric=='margin_proxy_growth_pct' else r.iloc[0].top3_pct)
    if a.output_dir.exists(): raise FileExistsError(f'Output directory exists; refusing to overwrite: {a.output_dir}')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    summary.to_csv(a.output_dir/'local_metric_summary.csv',index=False)
    comparison.to_csv(a.output_dir/'paper_comparison.csv',index=False)
    boot.to_csv(a.output_dir/'asin_cluster_bootstrap_ci.csv',index=False)
    audit.to_csv(a.output_dir/'dataset_integrity_summary.csv',index=False)
    if ranking is not None: ranking.to_csv(a.output_dir/'topk_comparison.csv',index=False)
    if robustness is not None: robustness.to_csv(a.output_dir/'robustness_comparison.csv',index=False)
    method={'paper':PAPER,'test_rows':len(test),'test_unique_ASINs':int(test.ASIN.nunique()),'test_months':months,'asin_clusters_for_bootstrap':nclusters,'bootstrap_replicates':a.bootstrap_replicates,'seed':a.seed,'inputs':{'evaluation_csv':str(a.evaluation_csv),'dataset_parquet':str(a.dataset_parquet),'product_metrics_csv':str(a.product_metrics_csv),'robustness_csv':str(a.robustness_csv)},'limitations':['Local margin outcome is a gross-margin proxy, not accounting profit.','Current environment is one-step; sequential cumulative reward is not calculated.','Paper reports 6810 test samples; local test count is not forcibly downsampled to match.','Top-k rankings are limited to the methods in the provided product metrics CSV.','ASIN-month duplicate records share the same growth target across differing discounts; this does not identify causal discount response.','Missing months and sparse ASIN timelines prevent assuming uninterrupted trajectories.'],'no_training_performed':True,'existing_artifacts_modified':False}
    (a.output_dir/'methodology.json').write_text(json.dumps(method,indent=2),encoding='utf-8')
    lines=['# Paper-comparable evaluation v2','',f"Base paper: {PAPER['title']} (IEEE Access 2026; DOI {PAPER['doi']}).",'', '## Test-set integrity','',f'- Test rows: {len(test):,}',f'- Unique ASINs: {test.ASIN.nunique():,}',f'- Mapped test months: {months}',f'- ASIN clusters used for bootstrap: {nclusters:,}', '- Source-row-to-product mapping was checked against `products_clean.parquet`; the script stops if mapped months are not exactly 9 and 10.','', '## Local metrics','',md_table(summary),'','## ASIN-cluster bootstrap 95% intervals','',md_table(boot) if len(boot) else 'Not available: fewer than two ASIN clusters.','', '## Paper comparison','',md_table(comparison),'', '## Dataset integrity','',md_table(audit),'']
    if ranking is not None: lines += ['## Top-1 / Top-3 rankings','',md_table(ranking),'','Rankings use only the methods present in the input file and minimum rank for ties.','']
    if robustness is not None: lines += ['## Existing prediction-noise robustness','',md_table(robustness),'']
    lines += ['## Interpretation safeguards','', '- These are simulator-derived outcomes, not observed causal market effects.','- The local margin field is a proxy, not verified accounting profit.','- Summed one-step rewards are explicitly not labelled sequential cumulative reward.','- The dataset contains multiple discount rows for many ASIN-months with invariant growth targets; discount elasticity is not identified by these records alone.','- The script performs no training and does not modify existing data, checkpoints, or reports.','']
    (a.output_dir/'paper_comparison_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Created: {a.output_dir}')
    print(f'Test rows={len(test):,}; unique ASINs={test.ASIN.nunique():,}; mapped months={months}')
    print(summary[['metric','value','unit']].to_string(index=False))
    print('ASIN-cluster bootstrap CIs saved:',a.output_dir/'asin_cluster_bootstrap_ci.csv')
    print('No model trained; no existing artifact overwritten.')

if __name__=='__main__': main()
