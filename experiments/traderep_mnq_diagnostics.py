from __future__ import annotations

import json
import numpy as np
import pandas as pd

from experiments.chain_engine import load_all, detect_breakdowns, find_fvg_not_mss
from experiments.unified_engine import UnifiedZone
from experiments.unified_engine_1m import build_5m_to_1m_map, reindex_to_1m


def main():
    d5 = load_all()
    nq1 = pd.read_parquet('data/NQ_1min_10yr.parquet')

    params = d5['params']
    nq5 = d5['nq']
    n5 = d5['n']
    o5, h5, l5, c5 = d5['o'], d5['h'], d5['l'], d5['c']
    atr5 = d5['atr_arr']
    bias5 = d5['bias_dir_arr']
    fvg_df = d5['fvg_df']
    on_hi5, on_lo5 = d5['on_hi'], d5['on_lo']
    swing_high_mask_5m = d5['swing_high_mask']
    swing_low_mask_5m = d5['swing_low_mask']
    swing_high_price_5m = d5['swing_high_price_at_mask']
    swing_low_price_5m = d5['swing_low_price_at_mask']

    fvg_bm = fvg_df['fvg_bull'].values
    fvg_bt = fvg_df['fvg_bull_top'].values
    fvg_bb = fvg_df['fvg_bull_bottom'].values
    fvg_em = fvg_df['fvg_bear'].values
    fvg_et = fvg_df['fvg_bear_top'].values
    fvg_eb = fvg_df['fvg_bear_bottom'].values

    all_bds_raw=[]
    for level_arr, level_type in [(on_lo5,'low'),(on_hi5,'high')]:
        all_bds_raw.extend(detect_breakdowns(h5,l5,c5,level_arr,level_type,
            min_depth_pts=params.get('chain',{}).get('min_depth_pts',1.0)))
    all_bds_raw.sort(key=lambda x:x['bar_idx'])
    all_bds=[]; last=-10
    for bd in all_bds_raw:
        if bd['bar_idx']-last >= 3:
            all_bds.append(bd); last=bd['bar_idx']

    chain={}; territory=set(); n_chain=0
    for bd in all_bds:
        bi=bd['bar_idx']
        for b in range(bi,min(bi+31,n5)): territory.add(b)
        zone=find_fvg_not_mss(fvg_bm,fvg_bt,fvg_bb,fvg_em,fvg_et,fvg_eb,
            h5,l5,c5,o5,atr5,swing_high_mask_5m,swing_low_mask_5m,
            swing_high_price_5m,swing_low_price_5m,bi,bd['level_type'],30,0.3)
        if zone is None: continue
        bd_atr=atr5[bi] if not np.isnan(atr5[bi]) else 30.0
        sweep_range=h5[bi]-l5[bi]
        uz=UnifiedZone(zone.direction,zone.top,zone.bottom,zone.size,zone.birth_bar,zone.birth_atr,1,
                       sweep_range/bd_atr if bd_atr>0 else 0)
        chain.setdefault(uz.birth_bar,[]).append(uz); n_chain+=1

    trend={}; n_trend=0
    for i in range(n5):
        for is_bull in (True,False):
            mask=fvg_bm if is_bull else fvg_em
            if not mask[i] or i in territory: continue
            direction=1 if is_bull else -1
            top=fvg_bt[i] if is_bull else fvg_et[i]
            bottom=fvg_bb[i] if is_bull else fvg_eb[i]
            size=top-bottom
            atr_val=atr5[i] if not np.isnan(atr5[i]) else 30.0
            if atr_val>0 and size < 0.3*atr_val: continue
            on_h,on_l=on_hi5[i],on_lo5[i]
            on_range=on_h-on_l if not(np.isnan(on_h) or np.isnan(on_l)) else 0
            if on_range<=0: continue
            mid=(top+bottom)/2; on_pos=(mid-on_l)/on_range
            if direction==1 and on_pos>=0.5: continue
            if direction==-1 and on_pos<0.5: continue
            bias=bias5[i]
            if direction==1 and bias<0: continue
            if direction==-1 and bias>0: continue
            uz=UnifiedZone(direction,top,bottom,size,i,atr_val,2)
            trend.setdefault(i,[]).append(uz); n_trend+=1

    map15=build_5m_to_1m_map(nq5,nq1)
    bias1=reindex_to_1m(bias5,map15)
    on_hi1=reindex_to_1m(on_hi5,map15); on_lo1=reindex_to_1m(on_lo5,map15)
    atr1=reindex_to_1m(atr5,map15)
    et=nq1.index.tz_convert('US/Eastern')
    etfrac=np.array([x.hour+x.minute/60 for x in et])
    h1=nq1['high'].values; l1=nq1['low'].values; c1=nq1['close'].values

    counters={k:0 for k in ['activated','survive_checks','eligible_time_zone_checks','touch','min_stop_fail','bias_fail','pd_fail','accepted','invalidated_close','expired']}
    tier_accept={1:0,2:0}; stop_dists=[]; zone_sizes=[]; local_atrs=[]
    active=[]; last5=-1
    max_age=200
    for i1 in range(len(nq1)):
        i5=map15[i1]
        if i5>last5:
            for b in range(last5+1,i5+1):
                zs=chain.get(b,[])+trend.get(b,[])
                active.extend(zs); counters['activated']+=len(zs)
            last5=i5
        surv=[]
        for z in active:
            if z.used: continue
            if (i5-z.birth_bar)>max_age:
                counters['expired']+=1; continue
            if z.direction==1 and c1[i1]<z.bottom:
                counters['invalidated_close']+=1; continue
            if z.direction==-1 and c1[i1]>z.top:
                counters['invalidated_close']+=1; continue
            surv.append(z)
        active=surv[-50:] if len(surv)>50 else surv
        if not (10.0 <= etfrac[i1] < 16.0): continue
        for z in active:
            if z.birth_bar>=i5: continue
            counters['eligible_time_zone_checks']+=1
            ep=z.top if z.direction==1 else z.bottom
            if z.direction==1 and l1[i1]>ep: continue
            if z.direction==-1 and h1[i1]<ep: continue
            counters['touch']+=1
            local_atr=atr1[i1] if not np.isnan(atr1[i1]) else 30.0
            sp=(z.bottom-z.size*0.15) if z.direction==1 else (z.top+z.size*0.15)
            sd=abs(ep-sp)
            sp=ep-sd*0.85 if z.direction==1 else ep+sd*0.85
            sd=abs(ep-sp); min_stop=0.15*local_atr
            stop_dists.append(float(sd)); zone_sizes.append(float(z.size)); local_atrs.append(float(local_atr))
            if sd<min_stop or sd<0.5:
                counters['min_stop_fail']+=1; continue
            if z.direction==1 and bias1[i1]<0:
                counters['bias_fail']+=1; continue
            if z.direction==-1 and bias1[i1]>0:
                counters['bias_fail']+=1; continue
            if z.tier==2:
                oh,ol=on_hi1[i1],on_lo1[i1]
                rng=oh-ol if not(np.isnan(oh) or np.isnan(ol)) else 0
                if rng>0:
                    fp=(ep-ol)/rng
                    if (z.direction==1 and fp>=0.5) or (z.direction==-1 and fp<0.5):
                        counters['pd_fail']+=1; continue
            counters['accepted']+=1; tier_accept[z.tier]+=1
            # Mark only in diagnostics to avoid repeatedly counting same would-be order.
            z.used=True
        counters['survive_checks']+=len(active)

    def q(a):
        if not a: return None
        x=np.array(a,dtype=float)
        return {'min':float(np.nanmin(x)),'p25':float(np.nanpercentile(x,25)),'median':float(np.nanmedian(x)),'p75':float(np.nanpercentile(x,75)),'max':float(np.nanmax(x))}
    out={'zones':{'chain':n_chain,'trend':n_trend,'breakdowns':len(all_bds)},'counters':counters,'accepted_by_tier':tier_accept,
         'stop_dist_pts':q(stop_dists),'zone_size_pts':q(zone_sizes),'local_atr_pts':q(local_atrs)}
    print('TRADEREP_DIAGNOSTICS',json.dumps(out,indent=2))
    with open('traderep_mnq_diagnostics.json','w') as f: json.dump(out,f,indent=2)

if __name__=='__main__': main()
