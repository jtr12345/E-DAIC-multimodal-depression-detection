"""E-DAIC five-fold CV with leakage-safe inner tuning of Logistic Regression C."""
from pathlib import Path
import argparse,json,warnings
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix
warnings.filterwarnings('ignore')

def labels(root):
    r=root/'labels' if (root/'labels').exists() else root; ds=[]
    for s in ('train','dev','test'):
        d=pd.read_csv(r/f'{s}_split.csv'); d['official_split']=s; ds.append(d)
    z=pd.concat(ds).drop_duplicates('Participant_ID').set_index('Participant_ID'); z.index=z.index.astype(str); return z
def agg(p,sep=','):
    xs=[]
    for x in pd.read_csv(p,sep=sep,chunksize=200000,low_memory=False):
        x=x.select_dtypes(include=[np.number]); x=x.drop(columns=[c for c in ('frame','timestamp','frameTime','confidence','success') if c in x],errors='ignore'); xs.append(x)
    x=pd.concat(xs,ignore_index=True); return pd.concat([x.mean().add_suffix('_mean'),x.std().fillna(0).add_suffix('_std')])
def load(pid,root):
    d=(root/'data' if (root/'data').exists() else root)/f'{pid}_P'; t=pd.read_csv(d/f'{pid}_Transcript.csv')['Text'].fillna('').astype(str).str.cat(sep=' '); return t,agg(d/'features'/f'{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'),agg(d/'features'/f'{pid}_OpenSMILE2.3.0_egemaps.csv',';')
def met(y,p,q): return {'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'auroc':float(roc_auc_score(y,q)),'confusion_matrix':confusion_matrix(y,p).tolist(),'n':int(len(y))}
def tune(x,y,seed):
    cv=StratifiedKFold(3,shuffle=True,random_state=seed); g=GridSearchCV(LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed),{'C':[0.01,0.1,1,3,10]},scoring='roc_auc',cv=cv,n_jobs=-1); g.fit(x,y); return g
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('edaic_experiment/result_3')); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    z=labels(a.root); ids=z.index.to_numpy(); y=z.PHQ_Binary.astype(int).to_numpy(); rec=[]
    for i,pid in enumerate(ids,1): print(f'loading {i}/{len(ids)} participant {pid}',flush=True); rec.append(load(pid,a.root))
    texts=[r[0] for r in rec]; V=pd.DataFrame([r[1] for r in rec],index=ids).replace([np.inf,-np.inf],np.nan).fillna(0); A=pd.DataFrame([r[2] for r in rec],index=ids).replace([np.inf,-np.inf],np.nan).fillna(0)
    skf=StratifiedKFold(5,shuffle=True,random_state=42); names=('text','audio','video','early_fusion','late_fusion'); pred={n:np.zeros(len(y),int) for n in names}; prob={n:np.zeros(len(y)) for n in names}; fid=np.zeros(len(y),int); rows=[]; chosen=[]
    for fold,(tr,va) in enumerate(skf.split(np.zeros(len(y)),y),1):
        fid[va]=fold; vec=TfidfVectorizer(max_features=3000,ngram_range=(1,2),min_df=2,sublinear_tf=True); xt=vec.fit_transform(np.array(texts)[tr]); xv=vec.transform(np.array(texts)[va]); svd=TruncatedSVD(n_components=min(50,max(2,xt.shape[1]-1)),random_state=42); xt=svd.fit_transform(xt); xv=svd.transform(xv); sa=StandardScaler().fit(A.iloc[tr]); sv=StandardScaler().fit(V.iloc[tr]); at=sa.transform(A.iloc[tr]); av=sa.transform(A.iloc[va]); vt=sv.transform(V.iloc[tr]); vv=sv.transform(V.iloc[va]); X={'text':(xt,xv),'audio':(at,av),'video':(vt,vv),'early_fusion':(np.hstack([xt,at,vt]),np.hstack([xv,av,vv]))}; fp={}
        for n,(tx,vx) in X.items():
            m=tune(tx,y[tr],42+fold); fp[n]=m.predict_proba(vx)[:,1]; prob[n][va]=fp[n]; pred[n][va]=(fp[n]>=.5).astype(int); chosen.append({'fold':fold,'model':n,'C':m.best_params_['C'],'inner_auc':m.best_score_})
        fp['late_fusion']=np.mean([fp[k] for k in ('text','audio','video')],axis=0); prob['late_fusion'][va]=fp['late_fusion']; pred['late_fusion'][va]=(fp['late_fusion']>=.5).astype(int)
        for n in names: rows.append({'fold':fold,'model':n,**met(y[va],pred[n][va],prob[n][va])})
        print(f'fold {fold}/5 complete',flush=True)
    res={n:met(y,pred[n],prob[n]) for n in names}; pd.DataFrame({'participant_id':ids,'label':y,'fold':fid,**{n+'_prob':prob[n] for n in names},**{n+'_pred':pred[n] for n in names}}).to_csv(a.out/'oof_predictions.csv',index=False); pd.DataFrame(rows).to_csv(a.out/'fold_metrics.csv',index=False); pd.DataFrame(chosen).to_csv(a.out/'chosen_C.csv',index=False); (a.out/'metrics.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2)); print('saved',a.out)
if __name__=='__main__': main()
