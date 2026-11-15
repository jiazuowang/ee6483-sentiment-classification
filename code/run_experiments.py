"""Reproducible EE6483 product-review sentiment experiments. No test labels used."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='2'
import argparse, json, time, hashlib, platform, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn, joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def features(kind):
    opts=dict(lowercase=True,strip_accents='unicode',sublinear_tf=True,min_df=2,dtype=np.float64)
    word=TfidfVectorizer(ngram_range=(1,1) if kind=='uni' else (1,2),max_features=2000 if kind=='neural' else 30000,**opts)
    if kind=='combined':
        char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=40000,**opts)
        return FeatureUnion([('word',word),('char',char)],transformer_weights={'word':1.0,'char':0.5})
    return word

def candidates():
    out=[]
    def add(name,kind,model,extra=None):
        steps=[('features',features(kind))]+(extra or [])+[('classifier',model)]
        out.append((name,Pipeline(steps)))
    add('NB_unigram_alpha1','uni',MultinomialNB(alpha=1.0))
    for c in [0.5,2.0,8.0]: add(f'LR_word_C{c:g}','word',LogisticRegression(C=c,max_iter=1500,solver='liblinear',random_state=42))
    add('SVM_unigram_C1','uni',LinearSVC(C=1.0,dual=True,max_iter=10000,random_state=42))
    for c in [0.25,1.0,4.0]: add(f'SVM_word_C{c:g}','word',LinearSVC(C=c,dual=True,max_iter=10000,random_state=42))
    for c in [0.5,2.0]: add(f'SVM_word_char_C{c:g}','combined',LinearSVC(C=c,dual=True,max_iter=10000,random_state=42))
    for c in [0.25,1.0,4.0]:
        add(f'SVM_word_balanced_C{c:g}','word',LinearSVC(C=c,class_weight='balanced',dual=True,max_iter=10000,random_state=42))
        add(f'SVM_word_char_balanced_C{c:g}','combined',LinearSVC(C=c,class_weight='balanced',dual=True,max_iter=10000,random_state=42))
    add('LR_word_balanced_C2','word',LogisticRegression(C=2,class_weight='balanced',max_iter=1500,solver='liblinear',random_state=42))
    for lr in [0.001,0.003]:
        add(f'MLP_TFIDF2000_lr{lr}','neural',MLPClassifier(hidden_layer_sizes=(32,),activation='relu',solver='adam',alpha=0.01,batch_size=128,learning_rate_init=lr,max_iter=150,early_stopping=True,validation_fraction=0.15,n_iter_no_change=15,random_state=42))
    return out

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'negative_f1':float(f1_score(y,p,pos_label=0)),'positive_f1':float(f1_score(y,p,pos_label=1))}

def main():
    required = [ROOT/'data'/name for name in ('train.json', 'test.json', 'submission_example.csv')]
    if any(not p.is_file() for p in required):
        raise SystemExit('Course data are not distributed. See data/README.md for the required local inputs.')
    res=ROOT/'results'; fig=ROOT/'figures';res.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
    train=pd.read_json(ROOT/'data/train.json');test=pd.read_json(ROOT/'data/test.json');template=pd.read_csv(ROOT/'data/submission_example.csv',keep_default_na=False)
    assert list(train.columns)==['reviews','sentiments'] and 'sentiments' not in test.columns
    assert train.reviews.map(lambda x:isinstance(x,str)).all() and test.reviews.map(lambda x:isinstance(x,str)).all()
    assert not train.isna().any().any() and not test.isna().any().any()
    assert set(train.sentiments)=={0,1} and len(test)==len(template) and template.reviews.tolist()==test.reviews.tolist()
    norm=lambda s:s.str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
    canonical=norm(train.reviews); test_canonical=norm(test.reviews); groups=pd.factorize(canonical)[0]
    x=train.reviews.to_numpy();y=train.sentiments.to_numpy(dtype=int)
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    dev,hold=next(outer.split(x,y,groups));assert set(groups[dev]).isdisjoint(groups[hold])
    pd.DataFrame({'row_index':np.arange(len(train)),'split':np.where(np.isin(np.arange(len(train)),hold),'holdout','development'),'group':groups}).to_csv(res/'split_assignments.csv',index=False)
    labelsets=train.assign(canonical=canonical).groupby('canonical').sentiments.nunique()
    audit={'train_rows':len(train),'test_rows':len(test),'train_class_counts':{str(k):int(v) for k,v in train.sentiments.value_counts().items()},'empty_train':int((canonical=='').sum()),'empty_test':int((test_canonical=='').sum()),'exact_duplicate_train_rows':int(train.reviews.duplicated().sum()),'normalized_duplicate_train_rows':int(canonical.duplicated().sum()),'conflicting_label_groups':int((labelsets>1).sum()),'test_overlap_with_train_rows':int(test_canonical.isin(set(canonical)).sum()),'development_rows':len(dev),'holdout_rows':len(hold),'development_counts':{str(k):int(v) for k,v in pd.Series(y[dev]).value_counts().items()},'holdout_counts':{str(k):int(v) for k,v in pd.Series(y[hold]).value_counts().items()},'review_word_length_quantiles':{str(k):float(v) for k,v in train.reviews.str.split().str.len().quantile([0,.5,.9,.99,1]).items()},'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').iterdir() if p.is_file()}}
    (res/'data_audit.json').write_text(json.dumps(audit,indent=2))
    inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026).split(x[dev],y[dev],groups[dev]))
    records=pd.read_csv(res/'cv_folds.csv').to_dict('records') if '--resume' in sys.argv and (res/'cv_folds.csv').exists() else [];allwarnings=[];configs=candidates();start=time.perf_counter()
    for name,pipe in configs:
        for fold,(a,b) in enumerate(inner,1):
            if any(r['model']==name and r['fold']==fold for r in records):continue
            assert set(groups[dev][a]).isdisjoint(groups[dev][b])
            m=clone(pipe);t=time.perf_counter()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always');m.fit(x[dev][a],y[dev][a]);p=m.predict(x[dev][b])
                allwarnings += [{'model':name,'fold':fold,'message':str(w.message)} for w in ws]
            r={'model':name,'fold':fold,**metrics(y[dev][b],p),'seconds':time.perf_counter()-t};records.append(r)
            print(json.dumps(r),flush=True)
            pd.DataFrame(records).to_csv(res/'cv_folds.csv',index=False)
    cv=pd.DataFrame(records); summary=cv.groupby('model').agg(macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),mean_seconds=('seconds','mean')).sort_values(['macro_f1_mean','accuracy_mean'],ascending=False)
    summary.to_csv(res/'cv_summary.csv');winner=str(summary.index[0]);print('LOCKED WINNER',winner,flush=True)
    # Selection is locked before looking at the holdout labels.
    (res/'selection.json').write_text(json.dumps({'selected_model':winner,'criterion':'mean development 3-fold grouped CV macro F1; accuracy tie-break','holdout_used_for_selection':False},indent=2))
    models=dict(configs); chosen=clone(models[winner]); t=time.perf_counter();chosen.fit(x[dev],y[dev]);fitsec=time.perf_counter()-t
    pred=chosen.predict(x[hold]);score=chosen.decision_function(x[hold]) if hasattr(chosen,'decision_function') else chosen.predict_proba(x[hold])[:,1]-.5
    hp=pd.DataFrame({'row_index':hold,'reviews':x[hold],'truth':y[hold],'prediction':pred,'decision_score':score,'correct':pred==y[hold]});hp.to_csv(res/'holdout_predictions.csv',index=False)
    cm=confusion_matrix(y[hold],pred,labels=[0,1]); report={'selected_model':winner,**metrics(y[hold],pred),'confusion_matrix_labels':[0,1],'confusion_matrix':cm.tolist(),'classification_report':classification_report(y[hold],pred,output_dict=True),'holdout_fit_seconds':fitsec,'majority_baseline_accuracy':float(np.mean(y[hold]==pd.Series(y[dev]).mode()[0]))}
    rng=np.random.default_rng(2026); boots=[]
    hold_group_indices=[np.flatnonzero(groups[hold]==g) for g in np.unique(groups[hold])]
    for _ in range(2000):
        sampled=rng.integers(0,len(hold_group_indices),len(hold_group_indices));ii=np.concatenate([hold_group_indices[j] for j in sampled]);boots.append(accuracy_score(y[hold][ii],pred[ii]))
    report['accuracy_group_bootstrap_95ci']=np.quantile(boots,[.025,.975]).tolist()
    # MLP initialization sensitivity on the locked split, descriptive only.
    mlpname=next(n for n in summary.index if n.startswith('MLP'))
    seeds=[]
    for seed in [42,123,2026]:
        m=clone(models[mlpname]).set_params(classifier__random_state=seed)
        t=time.perf_counter();m.fit(x[dev],y[dev]);pr=m.predict(x[hold]);seeds.append({'model':mlpname,'seed':seed,**metrics(y[hold],pr),'seconds':time.perf_counter()-t,'epochs':int(m.named_steps['classifier'].n_iter_)})
    pd.DataFrame(seeds).to_csv(res/'mlp_seed_results.csv',index=False)
    # Resource/feature ablations use development CV outcomes, never test labels.
    resources=[]
    for name in ['SVM_unigram_C1','SVM_word_C1','SVM_word_balanced_C0.25','SVM_word_char_balanced_C0.25',mlpname]:
        m=clone(models[name]);t=time.perf_counter();m.fit(x[dev],y[dev]);elapsed=time.perf_counter()-t
        z=m[:-1].transform(x[dev]);nnz=int(z.nnz) if scipy.sparse.issparse(z) else int(np.count_nonzero(z));bytes_=z.data.nbytes+z.indices.nbytes+z.indptr.nbytes if scipy.sparse.issparse(z) else z.nbytes
        t=time.perf_counter();m.predict(x[hold]);pt=time.perf_counter()-t
        resources.append({'model':name,'rows':z.shape[0],'dimensions':z.shape[1],'nonzeros':nnz,'feature_MiB':bytes_/1024**2,'fit_seconds':elapsed,'predict_holdout_seconds':pt,'cv_macro_f1':float(summary.loc[name,'macro_f1_mean'])})
    pd.DataFrame(resources).to_csv(res/'resource_comparison.csv',index=False)
    t=time.perf_counter();final=clone(models[winner]).fit(x,y);testpred=final.predict(test.reviews.to_numpy());report['final_fit_seconds']=time.perf_counter()-t
    submission=pd.DataFrame({'reviews':test.reviews,'predictions':testpred.astype(int)});submission.to_csv(ROOT/'submission.csv',index=False)
    assert len(submission)==1851 and list(submission.columns)==['reviews','predictions'] and set(submission.predictions)<={0,1}
    assert submission.reviews.tolist()==test.reviews.tolist()
    joblib.dump(final,res/'sentiment_model.joblib',compress=3)
    report['final_test_prediction_counts']={str(k):int(v) for k,v in submission.predictions.value_counts().items()}
    report['final_feature_dimensions']=int(final[:-1].transform(x[:1]).shape[1]);report['current_invocation_elapsed_seconds']=time.perf_counter()-start
    report['versions']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__}
    if hasattr(final.named_steps['classifier'],'coef_'):
        names=final.named_steps['features'].get_feature_names_out();coef=final.named_steps['classifier'].coef_[0];order=np.argsort(coef)
        pd.DataFrame({'feature':names[np.r_[order[:25],order[-25:]]],'weight':coef[np.r_[order[:25],order[-25:]]]}).to_csv(res/'feature_weights.csv',index=False)
    (res/'evaluation.json').write_text(json.dumps(report,indent=2));(res/'warnings.json').write_text(json.dumps(allwarnings,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig1,ax=plt.subplots(figsize=(8,7));ss=summary.sort_values('macro_f1_mean');ax.barh(ss.index,ss.macro_f1_mean,xerr=ss.macro_f1_std,color='#315b7b',capsize=3);ax.set_xlabel('Development 3-fold CV macro F1 (mean and sample SD)');ax.set_xlim(0,1);fig1.tight_layout();fig1.savefig(fig/'cv_comparison.png',dpi=180);plt.close(fig1)
    fig1,ax=plt.subplots(figsize=(4.5,4));ax.imshow(cm,cmap='Blues');ax.set_xticks([0,1],['Negative','Positive']);ax.set_yticks([0,1],['Negative','Positive']);ax.set_xlabel('Predicted');ax.set_ylabel('True');
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=16)
    fig1.tight_layout();fig1.savefig(fig/'holdout_confusion.png',dpi=180);plt.close(fig1)
    fig1,axs=plt.subplots(1,2,figsize=(8,3));counts=train.sentiments.value_counts().sort_index();axs[0].bar(['Negative','Positive'],counts,color=['#bb6b50','#315b7b']);axs[0].set_ylabel('Training reviews');axs[1].hist(train.reviews.str.split().str.len(),bins=40,color='#315b7b');axs[1].set_xlabel('Whitespace-separated words');axs[1].set_ylabel('Reviews');fig1.tight_layout();fig1.savefig(fig/'data_profile.png',dpi=180);plt.close(fig1)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
