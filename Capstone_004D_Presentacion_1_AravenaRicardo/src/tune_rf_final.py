from __future__ import annotations
import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd,sklearn,matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,recall_score,precision_score,f1_score,accuracy_score,confusion_matrix,average_precision_score,roc_curve,precision_recall_curve
sys.path.insert(0,str(Path(__file__).resolve().parent))
import config as cfg
from baseline_models import load_only_2024,feature_list,build_variant_preprocessor
from preprocessing import split_train_valid
SEED=42; OUTR=cfg.REPORTS_DIR; OUTF=cfg.FIGURES_DIR/'tuning_2024'; OUTF.mkdir(parents=True,exist_ok=True)

def metrics(y,p,t=.5):
 yp=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel();r=recall_score(y,yp)
 return {'threshold':float(t),'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,yp)),'recall':float(r),'precision':float(precision_score(y,yp,zero_division=0)),'f1':float(f1_score(y,yp,zero_division=0)),'fnr':float(1-r),'accuracy':float(accuracy_score(y,yp)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def fit_variant(Xtr,Xva,ytr,yva,group,class_weight=None):
 feats=feature_list(False);pre=build_variant_preprocessor(group_cod_depe=group,include_etnia=False,scale_numeric=False)
 Ztr=pre.fit_transform(Xtr[feats]);Zva=pre.transform(Xva[feats])
 model=RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=10,min_samples_split=2,max_features='sqrt',class_weight=class_weight,random_state=SEED,n_jobs=-1)
 t=time.perf_counter();model.fit(Ztr,ytr);fit=time.perf_counter()-t;p=model.predict_proba(Zva)[:,1]
 return pre,model,p,metrics(yva,p,.5),fit

def run():
 df=load_only_2024();Xtr,Xva,ytr,yva=split_train_valid(df)
 tests=[]; cache={}
 for group in [False,True]:
  for cw in [None,'balanced_subsample']:
   name=('agrupado_3' if group else 'original_1_6')+('|balanced' if cw else '|none')
   pre,model,p,m,fit=fit_variant(Xtr,Xva,ytr,yva,group,cw);cache[name]=(pre,model,p)
   tests.append({'variant':name,'cod_depe':'agrupado_3' if group else 'original_1_6','class_weight':str(cw),'fit_seconds':fit,**m})
   print(name,{k:round(m[k],4) for k in ['roc_auc','balanced_accuracy','recall','precision','f1','fnr']},flush=True)
 d=pd.DataFrame(tests).sort_values(['roc_auc','balanced_accuracy','f1'],ascending=False);d.to_csv(OUTR/'rf_tuning_2024.csv',index=False)
 winner=d.iloc[0]['variant'];p=cache[winner][2]
 rows=[metrics(yva,p,float(t)) for t in np.round(np.arange(.25,.701,.01),2)];sw=pd.DataFrame(rows);sw.to_csv(OUTR/'rf_threshold_2024.csv',index=False)
 feasible=sw[sw.recall>=.80];chosen=(feasible if len(feasible) else sw).sort_values(['f1','balanced_accuracy','precision'],ascending=False).iloc[0].to_dict()
 # plots
 fig,ax=plt.subplots(figsize=(9,5))
 for c in ['recall','precision','f1','balanced_accuracy']:ax.plot(sw.threshold,sw[c],label=c)
 ax.axvline(chosen['threshold'],ls='--',label=f"elegido={chosen['threshold']:.2f}");ax.legend();ax.set_xlabel('Threshold');ax.set_ylabel('Métrica');ax.set_title('RF tuned — threshold (validación 2024)');fig.tight_layout();fig.savefig(OUTF/'threshold_sweep.png',dpi=160);plt.close(fig)
 fpr,tpr,_=roc_curve(yva,p);fig,ax=plt.subplots(figsize=(6,5));ax.plot(fpr,tpr,label=f"AUC={roc_auc_score(yva,p):.4f}");ax.plot([0,1],[0,1],'--');ax.legend();ax.set_title('ROC RF tuned');fig.tight_layout();fig.savefig(OUTF/'roc_rf_tuned.png',dpi=160);plt.close(fig)
 pr,re,_=precision_recall_curve(yva,p);fig,ax=plt.subplots(figsize=(6,5));ax.plot(re,pr,label=f"AP={average_precision_score(yva,p):.4f}");ax.legend();ax.set_title('Precision-Recall RF tuned');fig.tight_layout();fig.savefig(OUTF/'pr_rf_tuned.png',dpi=160);plt.close(fig)
 report={'scope':'SOLO 2024; 2025 NO leído','train_rows':int(len(ytr)),'validation_rows':int(len(yva)),'features':['NEM','QUINTIL_SE4','COD_DEPE','EDAD','GENERO','NACIONALIDAD'],'etnia':'excluida','tuning_note':'Búsqueda controlada: exploración previa mostró que aumentar max_depth 16/20/None empeoraba AUC; se consolidó max_depth=12,min_samples_leaf=10 y se comparó COD_DEPE original/agrupado y class_weight con 160 árboles.','winner_variant':winner,'winner_params':{'n_estimators':160,'max_depth':12,'min_samples_leaf':10,'min_samples_split':2,'max_features':'sqrt','class_weight':None if winner.endswith('|none') else 'balanced_subsample','random_state':42},'winner_threshold_0_5':d.iloc[0].to_dict(),'threshold_selected_on_validation_2024':chosen,'threshold_rule':'max F1 sujeto a recall>=0.80; desempate balanced_accuracy/precision','metas':{'roc_auc':.80,'balanced_accuracy':.75,'recall':.80,'f1':.75,'fnr_max':.20},'cumplimiento_threshold_elegido':{'roc_auc':chosen['roc_auc']>=.80,'balanced_accuracy':chosen['balanced_accuracy']>=.75,'recall':chosen['recall']>=.80,'f1':chosen['f1']>=.75,'fnr':chosen['fnr']<=.20},'all_variants':d.to_dict(orient='records'),'versions':{'python':sys.version.split()[0],'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},'next_step':'Congelar hiperparámetros y threshold; evaluar una sola vez en 2025 solo si el usuario lo aprueba.'}
 with open(OUTR/'rf_tuning_2024.json','w',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
 print('\nFINAL\n'+json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':run()
