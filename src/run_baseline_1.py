"""Official E-DAIC split baseline using already extracted official features.

Text = TF-IDF; Audio = OpenSMILE eGeMAPS mean/std; Video = OpenFace
AU/pose/gaze mean/std. No raw data is copied or uploaded.
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
warnings.filterwarnings('ignore')

def load_labels(root):
    lab = root/'labels' if (root/'labels').exists() else root
    rows=[]
    for split in ('train','dev','test'):
        d=pd.read_csv(lab/f'{split}_split.csv')
        d['split']=split; rows.append(d)
    out=pd.concat(rows,ignore_index=True).set_index('Participant_ID')
    out.index=out.index.astype(str)
    return out

def agg_csv(path, sep=','):
    # Aggregate numeric frame-level features without retaining all frames.
    chunks=[]
    for c in pd.read_csv(path, sep=sep, chunksize=200000, low_memory=False):
        c=c.select_dtypes(include=[np.number])
        for x in ('frame','timestamp','frameTime','confidence','success'):
            if x in c: c=c.drop(columns=x)
        chunks.append(c)
    x=pd.concat(chunks,ignore_index=True)
    return pd.concat([x.mean().add_suffix('_mean'), x.std().fillna(0).add_suffix('_std')])

def load_one(pid, root, patch_root=None):
    d=(root/'data' if (root/'data').exists() else root)/f'{pid}_P'
    pd_=((patch_root/'data' if patch_root and (patch_root/'data').exists() else patch_root)/f'{pid}_P') if patch_root else None
    def locate(rel):
        p=pd_/rel if pd_ and (pd_/rel).exists() else d/rel
        return p
    tp=locate(f'{pid}_Transcript.csv')
    text=(pd.read_csv(tp).get('Text',pd.Series(dtype=str)).fillna('').astype(str).str.cat(sep=' ')
          if tp.exists() else '')
    vp=locate(f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv')
    v=agg_csv(vp) if vp.exists() else pd.Series(dtype=float)
    a=agg_csv(locate(f'features/{pid}_OpenSMILE2.3.0_egemaps.csv'),sep=';')
    return text,v,a

def score(y,p):
    out={'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),
         'precision':precision_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),
         'confusion_matrix':confusion_matrix(y,p).tolist()}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--patch-root',type=Path); ap.add_argument('--out',type=Path,default=Path('edaic_experiment/results')); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    labels=load_labels(a.root); ids=labels.index.astype(str)
    records=[]
    for i,pid in enumerate(ids,1):
        print(f'loading {i}/{len(ids)} participant {pid}',flush=True)
        t,v,au=load_one(pid,a.root,a.patch_root); records.append({'pid':pid,'text':t,'v':v,'a':au})
    V=pd.DataFrame([r['v'] for r in records],index=ids).replace([np.inf,-np.inf],np.nan).fillna(0)
    A=pd.DataFrame([r['a'] for r in records],index=ids).replace([np.inf,-np.inf],np.nan).fillna(0)
    text=[r['text'] for r in records]; y=labels.loc[ids,'PHQ_Binary'].astype(int).to_numpy(); split=labels.loc[ids,'split'].to_numpy()
    tr,dev,te=(split=='train'),(split=='dev'),(split=='test')
    vec=TfidfVectorizer(max_features=3000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
    Xt=vec.fit_transform(np.array(text)[tr]); Xd=vec.transform(np.array(text)[dev]); Xe=vec.transform(np.array(text)[te])
    svd=TruncatedSVD(n_components=min(50,max(2,Xt.shape[1]-1)),random_state=42); Xt2=svd.fit_transform(Xt); Xd2=svd.transform(Xd); Xe2=svd.transform(Xe)
    scalerV=StandardScaler().fit(V.loc[ids[tr]]); scalerA=StandardScaler().fit(A.loc[ids[tr]])
    Xv=scalerV.transform(V); Xa=scalerA.transform(A); Xearly=np.hstack([np.vstack([Xt2,Xd2,Xe2]),Xv,Xa])
    models={}
    def fit_predict(name,X):
        m=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42); m.fit(X[tr],y[tr]); pred=m.predict(X); prob=m.predict_proba(X)[:,1]; models[name]=(pred,prob); return m
    fit_predict('text',np.vstack([Xt2,Xd2,Xe2])); fit_predict('audio',Xa); fit_predict('video',Xv); fit_predict('early_fusion',Xearly)
    prob=np.mean([models[k][1] for k in ('text','audio','video')],axis=0); pred=(prob>=0.5).astype(int); models['late_fusion']=(pred,prob)
    results={}
    for name,(pred,prob) in models.items():
        results[name]={s:{**score(y[idx],pred[idx]),'auroc':roc_auc_score(y[idx],prob[idx]) if len(np.unique(y[idx]))>1 else None,'n':int(idx.sum())} for s,idx in [('train',tr),('dev',dev),('test',te)]}
    pd.DataFrame({'participant_id':ids,'split':split,'y':y,**{f'{k}_prob':v[1] for k,v in models.items()},**{f'{k}_pred':v[0] for k,v in models.items()}}).to_csv(a.out/'predictions.csv',index=False)
    (a.out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8'); print(json.dumps(results,indent=2)); print('saved',a.out)
if __name__=='__main__': main()
