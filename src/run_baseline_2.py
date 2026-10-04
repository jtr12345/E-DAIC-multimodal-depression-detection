"""E-DAIC five-fold participant-level cross-validation.

Each participant is held out exactly once. Text TF-IDF/SVD, scalers, and all
classifiers are fitted inside each training fold to avoid validation leakage.
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
warnings.filterwarnings('ignore')

def load_labels(root):
    lab=root/'labels' if (root/'labels').exists() else root
    out=[]
    for s in ('train','dev','test'):
        d=pd.read_csv(lab/f'{s}_split.csv'); d['official_split']=s; out.append(d)
    z=pd.concat(out,ignore_index=True).drop_duplicates('Participant_ID').set_index('Participant_ID'); z.index=z.index.astype(str); return z

def agg_csv(path, sep=','):
    chunks=[]
    for c in pd.read_csv(path,sep=sep,chunksize=200000,low_memory=False):
        c=c.select_dtypes(include=[np.number])
        c=c.drop(columns=[x for x in ('frame','timestamp','frameTime','confidence','success') if x in c],errors='ignore')
        chunks.append(c)
    x=pd.concat(chunks,ignore_index=True)
    return pd.concat([x.mean().add_suffix('_mean'),x.std().fillna(0).add_suffix('_std')])

def load_one(pid, root):
    d=(root/'data' if (root/'data').exists() else root)/f'{pid}_P'
    text=pd.read_csv(d/f'{pid}_Transcript.csv')['Text'].fillna('').astype(str).str.cat(sep=' ')
    v=agg_csv(d/'features'/f'{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv')
    au=agg_csv(d/'features'/f'{pid}_OpenSMILE2.3.0_egemaps.csv',sep=';')
    return text,v,au

def metrics(y,p,prob):
    return {'balanced_accuracy':float(balanced_accuracy_score(y,p)), 'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
            'precision':float(precision_score(y,p,zero_division=0)), 'recall':float(recall_score(y,p,zero_division=0)),
            'auroc':float(roc_auc_score(y,prob)), 'confusion_matrix':confusion_matrix(y,p).tolist(), 'n':int(len(y))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('edaic_experiment/result_2')); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    labels=load_labels(a.root); ids=labels.index.to_numpy(); y=labels['PHQ_Binary'].astype(int).to_numpy()
    records=[]
    for i,pid in enumerate(ids,1): print(f'loading {i}/{len(ids)} participant {pid}',flush=True); t,v,au=load_one(pid,a.root); records.append((t,v,au))
    texts=[x[0] for x in records]; V=pd.DataFrame([x[1] for x in records],index=ids).replace([np.inf,-np.inf],np.nan).fillna(0); A=pd.DataFrame([x[2] for x in records],index=ids).replace([np.inf,-np.inf],np.nan).fillna(0)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); names=('text','audio','video','early_fusion','late_fusion'); pred={n:np.zeros(len(ids),dtype=int) for n in names}; prob={n:np.zeros(len(ids),dtype=float) for n in names}; fold_id=np.zeros(len(ids),dtype=int); fold_rows=[]
    for fold,(tr,va) in enumerate(skf.split(np.zeros(len(y)),y),1):
        fold_id[va]=fold; print(f'fold {fold}/5 train={len(tr)} validation={len(va)}',flush=True)
        vec=TfidfVectorizer(max_features=3000,ngram_range=(1,2),min_df=2,sublinear_tf=True); Xt=vec.fit_transform(np.array(texts)[tr]); Xv=vec.transform(np.array(texts)[va]); ncomp=min(50,max(2,Xt.shape[1]-1)); svd=TruncatedSVD(n_components=ncomp,random_state=42); Xt=svd.fit_transform(Xt); Xv=svd.transform(Xv)
        sca=StandardScaler().fit(A.iloc[tr]); scv=StandardScaler().fit(V.iloc[tr]); At=sca.transform(A.iloc[tr]); Av=sca.transform(A.iloc[va]); Vt=scv.transform(V.iloc[tr]); Vv=scv.transform(V.iloc[va]);
        X={'text':(Xt,Xv),'audio':(At,Av),'video':(Vt,Vv),'early_fusion':(np.hstack([Xt,At,Vt]),np.hstack([Xv,Av,Vv]))}
        fold_probs={}
        for n,(tx,vx) in X.items():
            m=LogisticRegression(max_iter=1500,class_weight='balanced',C=1.0,random_state=42); m.fit(tx,y[tr]); fold_probs[n]=m.predict_proba(vx)[:,1]; prob[n][va]=fold_probs[n]; pred[n][va]=(fold_probs[n]>=0.5).astype(int)
        fold_probs['late_fusion']=np.mean([fold_probs[k] for k in ('text','audio','video')],axis=0); prob['late_fusion'][va]=fold_probs['late_fusion']; pred['late_fusion'][va]=(fold_probs['late_fusion']>=0.5).astype(int)
        for n in names: fold_rows.append({'fold':fold,'model':n,**metrics(y[va],pred[n][va],prob[n][va])})
    results={n:metrics(y,pred[n],prob[n]) for n in names}; pd.DataFrame({'participant_id':ids,'label':y,'fold':fold_id,**{n+'_prob':prob[n] for n in names},**{n+'_pred':pred[n] for n in names}}).to_csv(a.out/'oof_predictions.csv',index=False); pd.DataFrame(fold_rows).to_csv(a.out/'fold_metrics.csv',index=False); (a.out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8'); print(json.dumps(results,indent=2)); print('saved',a.out)
if __name__=='__main__': main()
