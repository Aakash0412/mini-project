#!/usr/bin/env python3
"""Read-only Top-K matched-cohort audit with ASIN-cluster bootstrap CIs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def args_parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--product-metrics-csv',type=Path,required=True)
    p.add_argument('--dataset-parquet',type=Path,default=Path('data/processed/products_clean.parquet'))
    p.add_argument('--output-dir',type=Path,default=Path('reports/topk_cohort_audit_v4'))
    p.add_argument('--bootstrap-replicates',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    return p.parse_args()

def require(df, cols, src):
    missing=[c for c in cols if c not in df.columns]
    if missing: raise ValueError(f'{src} missing required columns: {missing}')

def read_metrics(path):
    df=pd.read_csv(path)
    if {'source_row','algorithm','sales_growth_pct','margin_proxy_growth_pct'}.issubset(df.columns):
        x=df[['source_row','algorithm','sales_growth_pct','margin_proxy_growth_pct']].rename(columns={'algorithm':'method'}).copy(); schema='long_form_growth_metrics'
    elif {'source_row','method','sales_growth_pct','margin_proxy_growth_pct'}.issubset(df.columns):
        x=df[['source_row','method','sales_growth_pct','margin_proxy_growth_pct']].copy(); schema='long_form_growth_metrics'
    elif {'source_row_index','simulator','policy','baseline_sales','estimated_sales','baseline_margin_proxy','estimated_margin_proxy'}.issubset(df.columns):
        x=df[['source_row_index','simulator','policy','baseline_sales','estimated_sales','baseline_margin_proxy','estimated_margin_proxy']].rename(columns={'source_row_index':'source_row'}).copy()
        x['method']=x['simulator'].astype(str)+'::'+x['policy'].astype(str)
        for out,new,old in [('sales_growth_pct','estimated_sales','baseline_sales'),('margin_proxy_growth_pct','estimated_margin_proxy','baseline_margin_proxy')]:
            a=pd.to_numeric(x[new],errors='coerce'); b=pd.to_numeric(x[old],errors='coerce')
            x[out]=np.where(np.isfinite(a)&np.isfinite(b)&(b>0),(a/b-1)*100,np.nan)
        x=x[['source_row','method','sales_growth_pct','margin_proxy_growth_pct']]; schema='derived_from_simulator_policy_comparison'
    else:
        raise ValueError(f'Unsupported schema in {path}. Expected source_row + algorithm/method + sales_growth_pct + margin_proxy_growth_pct, or simulator/policy comparison columns.')
    x['source_row']=pd.to_numeric(x.source_row,errors='raise').astype(int); x['method']=x.method.astype(str).str.strip()
    for c in ['sales_growth_pct','margin_proxy_growth_pct']: x[c]=pd.to_numeric(x[c],errors='coerce')
    if (x.source_row<0).any(): raise ValueError('Negative source_row values found.')
    if x.duplicated(['source_row','method']).any(): raise ValueError('Duplicate source_row/method pairs found; resolve before ranking.')
    return x,schema

def attach_identity(x,path):
    if not path.exists(): raise FileNotFoundError(path)
    d=pd.read_parquet(path); require(d,['ASIN','MonthNum'],path)
    if x.source_row.max()>=len(d): raise ValueError('source_row outside dataset row-position range.')
    mapped=d.iloc[x.source_row.to_numpy()][['ASIN','MonthNum']].reset_index(drop=True)
    x=x.reset_index(drop=True).copy(); x['ASIN']=mapped.ASIN.astype(str); x['MonthNum']=pd.to_numeric(mapped.MonthNum,errors='coerce')
    months=sorted(x.MonthNum.dropna().astype(int).unique().tolist())
    if months != [9,10]: raise ValueError(f'Rows map to months {months}, not expected held-out months [9, 10].')
    return x,months

def rank_metric(x,metric):
    v=x.dropna(subset=[metric]).copy(); methods=sorted(v.method.unique())
    sets={m:set(v.loc[v.method.eq(m),'source_row']) for m in methods}
    common=set.intersection(*(sets[m] for m in methods)) if methods else set()
    c=v[v.source_row.isin(common)].copy()
    if c.empty: return pd.DataFrame(),c
    c['rank_min']=c.groupby('source_row')[metric].rank(method='min',ascending=False)
    c['rank_first']=c.groupby('source_row')[metric].rank(method='first',ascending=False)
    maxv=c.groupby('source_row')[metric].transform('max')
    n_top=c[metric].eq(maxv).groupby(c.source_row).transform('sum')
    c['strict_unique_top1']=c.rank_min.eq(1)&n_top.eq(1)
    ties=int((c.loc[c.rank_min.eq(1)].groupby('source_row').size()>1).sum())
    rows=[]
    for m in methods:
        g=c[c.method.eq(m)]; n=len(g)
        rows.append({'metric':metric,'method':m,'valid_rows_for_method_before_matching':int(sets[m].__len__()),'matched_source_rows':len(common),'matched_unique_ASINs':c[['source_row','ASIN']].drop_duplicates().ASIN.nunique(),'top1_minrank_count':int((g.rank_min==1).sum()),'top1_minrank_pct':100*(g.rank_min==1).mean(),'top1_strict_unique_winner_count':int(g.strict_unique_top1.sum()),'top1_strict_unique_winner_pct':100*g.strict_unique_top1.mean(),'top3_minrank_count':int((g.rank_min<=3).sum()),'top3_minrank_pct':100*(g.rank_min<=3).mean(),'top3_first_rank_count':int((g.rank_first<=3).sum()),'top3_first_rank_pct':100*(g.rank_first<=3).mean(),'source_rows_with_tied_top_value':ties,'methods_compared':len(methods)})
    return pd.DataFrame(rows),c

def bootstrap(c,metric,reps,seed):
    """Fast ASIN-cluster bootstrap using precomputed per-row ranks.

    Ranking is performed within each source row before bootstrapping. Resampling
    ASIN clusters changes observation weights, not the within-row method ranks,
    so ranks do not need to be recalculated in every replicate.
    """
    if c.empty:
        return pd.DataFrame()
    methods=sorted(c.method.unique())
    asins=pd.Index(c.ASIN.drop_duplicates(), name='ASIN')
    if len(asins)<2:
        return pd.DataFrame()
    rng=np.random.default_rng(seed)
    stat_specs={
        'top1_minrank':('rank_min',lambda s:s.eq(1)),
        'top1_strict_unique':('strict_unique_top1',lambda s:s.astype(bool)),
        'top3_minrank':('rank_min',lambda s:s.le(3)),
        'top3_first_rank':('rank_first',lambda s:s.le(3)),
    }
    # Cluster-level numerators and denominators; each cluster is resampled as a unit.
    aggregates={}
    for m in methods:
        g=c[c.method.eq(m)].copy()
        den=g.groupby('ASIN').size().reindex(asins,fill_value=0).to_numpy(dtype=float)
        aggregates[(m,'den')]=den
        for stat,(col,fn) in stat_specs.items():
            numer=fn(g[col]).astype(float).groupby(g['ASIN']).sum().reindex(asins,fill_value=0).to_numpy(dtype=float)
            aggregates[(m,stat)]=numer
    out=[]
    n_clusters=len(asins)
    for m in methods:
        for stat in stat_specs:
            estimates=np.empty(reps,dtype=float)
            num=aggregates[(m,stat)]
            den=aggregates[(m,'den')]
            for b in range(reps):
                draws=rng.integers(0,n_clusters,size=n_clusters)
                multiplicity=np.bincount(draws,minlength=n_clusters).astype(float)
                total_den=float(den @ multiplicity)
                estimates[b]=100.0*float(num @ multiplicity)/total_den if total_den>0 else np.nan
            estimates=estimates[np.isfinite(estimates)]
            out.append({'metric':metric,'method':m,'statistic':stat,'cluster_unit':'ASIN',
                'ASIN_clusters':n_clusters,'bootstrap_replicates_requested':reps,
                'valid_replicates':len(estimates),
                'ci95_low':float(np.percentile(estimates,2.5)) if len(estimates) else np.nan,
                'ci95_high':float(np.percentile(estimates,97.5)) if len(estimates) else np.nan,
                'bootstrap_mean':float(np.mean(estimates)) if len(estimates) else np.nan})
    return pd.DataFrame(out)

def main():
    a=args_parse()
    if a.bootstrap_replicates<100: raise ValueError('--bootstrap-replicates must be at least 100.')
    if a.output_dir.exists(): raise FileExistsError(f'Refusing to overwrite existing output directory: {a.output_dir}')
    x,schema=read_metrics(a.product_metrics_csv); x,months=attach_identity(x,a.dataset_parquet)
    ranking=[]; boot=[]; cohorts=[]
    for i,metric in enumerate(['margin_proxy_growth_pct','sales_growth_pct']):
        r,c=rank_metric(x,metric); ranking.append(r); z=bootstrap(c,metric,a.bootstrap_replicates,a.seed+i)
        if not z.empty: boot.append(z)
        cohorts.append({'metric':metric,'input_valid_rows':int(x[metric].notna().sum()),'methods':int(x.loc[x[metric].notna(),'method'].nunique()),'matched_source_rows':int(c.source_row.nunique()),'matched_unique_ASINs':int(c[['source_row','ASIN']].drop_duplicates().ASIN.nunique()) if len(c) else 0,'mapped_months':';'.join(map(str,months)),'one_row_per_source_row_method':bool(not c.duplicated(['source_row','method']).any()) if len(c) else True})
    ranks=pd.concat(ranking,ignore_index=True) if ranking else pd.DataFrame(); boots=pd.concat(boot,ignore_index=True) if boot else pd.DataFrame(); cohort_df=pd.DataFrame(cohorts)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    ranks.to_csv(a.output_dir/'topk_matched_cohort_metrics.csv',index=False); boots.to_csv(a.output_dir/'topk_asin_cluster_bootstrap_ci.csv',index=False); cohort_df.to_csv(a.output_dir/'matched_cohort_summary.csv',index=False)
    meta={'input_csv':str(a.product_metrics_csv),'dataset_parquet':str(a.dataset_parquet),'input_schema':schema,'mapped_months':months,'cohort_rule':'Intersection of valid source rows across every supplied method, computed separately for each metric.','rank_rules':{'top1_minrank':'Tie-inclusive; tied methods all count as Top-1.','top1_strict_unique_winner':'Only a unique maximum counts as a win; tied maxima count as no unique winner.','top3_minrank':'Tie-inclusive competition rank <= 3.','top3_first_rank':'Input-order tie-break sensitivity analysis.'},'paper_reference':{'test_samples':6810,'ppo_top1_profit_frequency_pct':52.32,'ppo_top3_sales_frequency_pct':74.33,'comparability_warning':'Context only; direct comparison requires same cohort, outcome definition, method set and tie protocol.'},'bootstrap':{'unit':'ASIN cluster','replicates':a.bootstrap_replicates,'seed':a.seed,'interval':'percentile 95%'},'limitations':['source_row must be zero-based row position in products_clean.parquet; held-out month check is enforced.','Margin is a proxy, not verified accounting profit.','Top-K values depend on methods supplied in input.','Matched cohort can differ from paper sample of 6810.','Bootstrap resamples ASINs, not independent training seeds.','No training; no input/checkpoint/report overwrite.'],'no_training_performed':True,'existing_artifacts_modified':False}
    (a.output_dir/'methodology.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    report=['# Top-K matched-cohort and ASIN-cluster bootstrap audit','','- Input schema: '+schema,'- Mapped months: '+str(months),'- Methods: '+', '.join(sorted(x.method.unique())),'- Bootstrap replicates: '+str(a.bootstrap_replicates),'','## Matched cohort','',cohort_df.to_markdown(index=False),'','## Top-1 / Top-3 results','',ranks.to_markdown(index=False),'','## ASIN-cluster bootstrap 95% confidence intervals','',boots.to_markdown(index=False) if not boots.empty else 'No intervals produced.','','## Interpretation safeguards','', '- Use strict unique-winner Top-1 to measure sole best method; tied maxima do not count as a win.','- Tie-inclusive Top-1/Top-3 percentages may sum to more than 100% across methods.','- Top3 first-rank is a tie-break sensitivity analysis and depends on input order.','- Do not claim direct paper comparability until cohort, metric, method set and tie rules are aligned.','- Simulated outcomes are not causal market effects; margin proxy is not accounting profit.','- No model was trained and no existing artifact was overwritten.','']
    (a.output_dir/'topk_matched_cohort_report.md').write_text('\n'.join(report),encoding='utf-8')
    print(f'Created: {a.output_dir}\nInput schema: {schema}; mapped months: {months}')
    print(cohort_df.to_string(index=False)); print('\nTop-K metrics:\n',ranks.to_string(index=False)); print('\nNo model trained; no existing artifact overwritten.')
if __name__=='__main__': main()
