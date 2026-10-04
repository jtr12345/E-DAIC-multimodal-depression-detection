"""Nested threshold selection for the strongest candidates from result_4."""
from pathlib import Path
import argparse,json,warnings
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix
warnings.filterwarnings('ignore')
from run_baseline_4 import labels,agg,load,met

def make_model(kind,seed=42,scale=1.0):
 if kind=='linear_svm': return LinearSVC(C=1.0,class_weight='balanced',random_state=seed)
 if kind=='random_forest': return RandomForestClassifier(n_estimators=300,max_features='sqrt',min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=seed)
 return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.8,min_child_weight=3,reg_lambda=5,reg_alpha=.1,scale_pos_weight=scale,objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=-1,random_state=seed)
def score_model(m,x): return m.decision_function(x) if isinstance(m,LinearSVC) else m.predict_proba(x)[:,1]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('edaic_experiment/result_5')); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 z=labels(a.root); ids=z.index.to_numpy(); y=z.PHQ_Binary.astype(int).to_numpy(); rec=[]
 for i,pid in enumerate(ids,1): print(f'loading {i}/{len(ids)} participant {pid}',flush=True); rec.append(load(pid,a.root))
 texts=[r[0] for r in rec]; V=pd.DataFrame([r[1] for r in rec],index=ids).replace([np.inf,-np.inf],np.nan).fillna(0); A=pd.DataFrame([r[2] for r in rec],index=ids).replace([np.inf,-np.inf],np.nan).fillna(0)
 candidates=(('linear_svm','text'),('random_forest','video'),('xgboost','early_fusion')); skf=StratifiedKFold(5,shuffle=True,random_state=42); outp={f'{m}_{n}':np.zeros(len(y)) for m,n in candidates}; outy={f'{m}_{n}':np.zeros(len(y),int) for m,n in candidates}; fid=np.zeros(len(y),int); rows=[]
 for fold,(tr,va) in enumerate(skf.split(np.zeros(len(y)),y),1):
  fid[va]=fold; vec=TfidfVectorizer(max_features=3000,ngram_range=(1,2),min_df=2,sublinear_tf=True); xt=vec.fit_transform(np.array(texts)[tr]); xv=vec.transform(np.array(texts)[va]); svd=TruncatedSVD(n_components=min(50,max(2,xt.shape[1]-1)),random_state=42); xt=svd.fit_transform(xt); xv=svd.transform(xv); sa=StandardScaler().fit(A.iloc[tr]); sv=StandardScaler().fit(V.iloc[tr]); at=sa.transform(A.iloc[tr]); av=sa.transform(A.iloc[va]); vt=sv.transform(V.iloc[tr]); vv=sv.transform(V.iloc[va]); X={'text':(xt,xv),'video':(vt,vv),'early_fusion':(np.hstack([xt,at,vt]),np.hstack([xv,av,vv]))}
  inner=StratifiedKFold(3,shuffle=True,random_state=100+fold)
  for kind,n in candidates:
   tx,vx=X[n]; inner_prob=np.zeros(len(tr)); scale=float((y[tr]==0).sum()/max(1,(y[tr]==1).sum())) if kind=='xgboost' else 1.0
   for itr,iva in inner.split(tx,y[tr]):
    m=make_model(kind,42+fold,scale); m.fit(tx[itr],y[tr][itr]); inner_prob[iva]=score_model(m,tx[iva])
   thresholds=np.arange(-0.5,0.71,0.05) if kind=='linear_svm' else np.arange(.2,.81,.05); best=max(thresholds,key=lambda q:f1_score(y[tr],(inner_prob>=q).astype(int),average='macro',zero_division=0))
   m=make_model(kind,42+fold,scale); m.fit(tx,y[tr]); q=score_model(m,vx); p=(q>=best).astype(int); key=f'{kind}_{n}'; outp[key][va]=q; outy[key][va]=p; rows.append({'fold':fold,'model':key,'threshold':float(best),'scale_pos_weight':scale,**met(y[va],p,q)})
  print(f'fold {fold}/5 complete',flush=True)
 res={k:met(y,outy[k],outp[k]) for k in outp}; pd.DataFrame({'participant_id':ids,'label':y,'fold':fid,**{k+'_score':outp[k] for k in outp},**{k+'_pred':outy[k] for k in outy}}).to_csv(a.out/'oof_predictions.csv',index=False); pd.DataFrame(rows).to_csv(a.out/'fold_metrics.csv',index=False); (a.out/'metrics.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2)); print('saved',a.out)
if __name__=='__main__': main()
