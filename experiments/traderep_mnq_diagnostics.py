from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np, pandas as pd
from experiments.chain_engine import load_all, detect_breakdowns, find_fvg_not_mss
from experiments.unified_engine import UnifiedZone
from experiments.unified_engine_1m import build_5m_to_1m_map, reindex_to_1m

def main():
 d5=load_all(); nq1=pd.read_parquet('data/NQ_1min_10yr.parquet'); nq5=d5['nq']; p=d5['params']; n5=d5['n']
 o5,h5,l5,c5=d5['o'],d5['h'],d5['l'],d5['c']; atr5=d5['atr_arr']; bias5=d5['bias_dir_arr']; f=d5['fvg_df']; on_hi5,on_lo5=d5['on_hi'],d5['on_lo']
 shm,slm,shp,slp=d5['swing_high_mask'],d5['swing_low_mask'],d5['swing_high_price_at_mask'],d5['swing_low_price_at_mask']
 bm,bt,bb=f['fvg_bull'].values,f['fvg_bull_top'].values,f['fvg_bull_bottom'].values; em,et,eb=f['fvg_bear'].values,f['fvg_bear_top'].values,f['fvg_bear_bottom'].values
 raw=[]
 for a,t in [(on_lo5,'low'),(on_hi5,'high')]: raw+=detect_breakdowns(h5,l5,c5,a,t,min_depth_pts=p.get('chain',{}).get('min_depth_pts',1.0))
 raw.sort(key=lambda x:x['bar_idx']); bds=[]; last=-10
 for x in raw:
  if x['bar_idx']-last>=3: bds.append(x); last=x['bar_idx']
 chain={}; terr=set(); nc=0
 for bd in bds:
  bi=bd['bar_idx']; terr.update(range(bi,min(bi+31,n5))); z=find_fvg_not_mss(bm,bt,bb,em,et,eb,h5,l5,c5,o5,atr5,shm,slm,shp,slp,bi,bd['level_type'],30,0.3)
  if z is None: continue
  a=atr5[bi] if not np.isnan(atr5[bi]) else 30.; sr=h5[bi]-l5[bi]; u=UnifiedZone(z.direction,z.top,z.bottom,z.size,z.birth_bar,z.birth_atr,1,sr/a if a>0 else 0); chain.setdefault(u.birth_bar,[]).append(u); nc+=1
 trend={}; nt=0
 for i in range(n5):
  for bull in (True,False):
   mask=bm if bull else em
   if not mask[i] or i in terr: continue
   d=1 if bull else -1; top=bt[i] if bull else et[i]; bot=bb[i] if bull else eb[i]; size=top-bot; a=atr5[i] if not np.isnan(atr5[i]) else 30.
   if a>0 and size<.3*a: continue
   oh,ol=on_hi5[i],on_lo5[i]; rng=oh-ol if not(np.isnan(oh) or np.isnan(ol)) else 0
   if rng<=0: continue
   pos=((top+bot)/2-ol)/rng
   if (d==1 and pos>=.5) or (d==-1 and pos<.5): continue
   if (d==1 and bias5[i]<0) or (d==-1 and bias5[i]>0): continue
   u=UnifiedZone(d,top,bot,size,i,a,2); trend.setdefault(i,[]).append(u); nt+=1
 m=build_5m_to_1m_map(nq5,nq1); bias1=reindex_to_1m(bias5,m); hi1=reindex_to_1m(on_hi5,m); lo1=reindex_to_1m(on_lo5,m); atr1=reindex_to_1m(atr5,m)
 news5=d5['news_blackout_arr']; news1=reindex_to_1m(news5,m) if news5 is not None else np.zeros(len(nq1),dtype=bool)
 e=nq1.index.tz_convert('US/Eastern'); ef=np.array([x.hour+x.minute/60 for x in e]); h1,l1,c1=nq1.high.values,nq1.low.values,nq1.close.values
 keys=['activated','touch','min_stop_fail','bias_fail','pd_fail','accepted','accepted_news_blocked','accepted_not_news','invalidated','expired']; C={k:0 for k in keys}; tiers={1:0,2:0}; active=[]; last=-1
 for j in range(len(nq1)):
  i=m[j]
  if i>last:
   for b in range(last+1,i+1):
    zs=chain.get(b,[])+trend.get(b,[]); active+=zs; C['activated']+=len(zs)
   last=i
  s=[]
  for z in active:
   if z.used: continue
   if i-z.birth_bar>200: C['expired']+=1; continue
   if (z.direction==1 and c1[j]<z.bottom) or (z.direction==-1 and c1[j]>z.top): C['invalidated']+=1; continue
   s.append(z)
  active=s[-50:]
  if not(10<=ef[j]<16): continue
  for z in active:
   if z.birth_bar>=i: continue
   ep=z.top if z.direction==1 else z.bottom
   if (z.direction==1 and l1[j]>ep) or (z.direction==-1 and h1[j]<ep): continue
   C['touch']+=1; a=atr1[j] if not np.isnan(atr1[j]) else 30.; sp=z.bottom-z.size*.15 if z.direction==1 else z.top+z.size*.15; sd=abs(ep-sp)*.85
   if sd<.15*a or sd<.5: C['min_stop_fail']+=1; continue
   if (z.direction==1 and bias1[j]<0) or (z.direction==-1 and bias1[j]>0): C['bias_fail']+=1; continue
   if z.tier==2:
    rng=hi1[j]-lo1[j] if not(np.isnan(hi1[j]) or np.isnan(lo1[j])) else 0
    if rng>0:
     fp=(ep-lo1[j])/rng
     if (z.direction==1 and fp>=.5) or (z.direction==-1 and fp<.5): C['pd_fail']+=1; continue
   C['accepted']+=1; tiers[z.tier]+=1
   if bool(news1[j]): C['accepted_news_blocked']+=1
   else: C['accepted_not_news']+=1
   z.used=True
 out={'zones':{'chain':nc,'trend':nt,'breakdowns':len(bds)},'counters':C,'accepted_by_tier':tiers,'news_blackout':{'bars':int(np.count_nonzero(news1)),'fraction':float(np.mean(news1.astype(bool)))}}
 print('TRADEREP_DIAGNOSTICS',json.dumps(out,indent=2)); open('traderep_mnq_diagnostics.json','w').write(json.dumps(out,indent=2))
if __name__=='__main__': main()
