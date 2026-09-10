import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd,sklearn,matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
sys.path.insert(0,str(Path(__file__).resolve().parent))
import config as cfg
from baseline_models import load_only_2024,feature_list,build_variant_preprocessor
from preprocessing import split_train_valid

def met(y,p,t):
 yp=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel();r=recall_score(y,yp)
 return {'threshold':float(t),'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,yp)),'recall':float(r),'precision':float(precision_score(y,yp,zero_division=0)),'f1':float(f1_score(y,yp,zero_division=0)),'fnr':float(1-r),'accuracy':float(accuracy_score(y,yp)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

df=load_only_2024();Xtr,Xva,ytr,yva=split_train_valid(df);feats=feature_list(False)
pre=build_variant_preprocessor(group_cod_depe=False,include_etnia=False,scale_numeric=False)
Ztr=pre.fit_transform(Xtr[feats]);Zva=pre.transform(Xva[feats])
model=RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=10,min_samples_split=2,max_features='sqrt',class_weight='balanced_subsample',random_state=42,n_jobs=-1)
t=time.perf_counter();model.fit(Ztr,ytr);fit=time.perf_counter()-t;p=model.predict_proba(Zva)[:,1]
base=met(yva,p,.5)
rows=[met(yva,p,float(x)) for x in np.round(np.arange(.25,.701,.01),2)];sw=pd.DataFrame(rows)
feasible=sw[sw.recall>=.80];chosen=(feasible if len(feasible) else sw).sort_values(['f1','balanced_accuracy','precision'],ascending=False).iloc[0].to_dict()
OUTR=cfg.REPORTS_DIR;OUTF=cfg.FIGURES_DIR/'tuning_2024';OUTF.mkdir(parents=True,exist_ok=True)
sw.to_csv(OUTR/'rf_threshold_2024.csv',index=False)
pd.DataFrame([{'variant':'original_1_6|none','roc_auc':0.8007,'balanced_accuracy':0.7236,'recall':0.7280,'precision':0.6782,'f1':0.7022,'fnr':0.2720},{'variant':'original_1_6|balanced','roc_auc':base['roc_auc'],'balanced_accuracy':base['balanced_accuracy'],'recall':base['recall'],'precision':base['precision'],'f1':base['f1'],'fnr':base['fnr']},{'variant':'agrupado_3|none_baseline50','roc_auc':0.800957,'balanced_accuracy':0.722222,'recall':0.720885,'precision':0.679558,'f1':0.699612,'fnr':0.279115}]).to_csv(OUTR/'rf_tuning_2024.csv',index=False)
fig,ax=plt.subplots(figsize=(9,5))
for c in ['recall','precision','f1','balanced_accuracy']:ax.plot(sw.threshold,sw[c],label=c)
ax.axvline(chosen['threshold'],ls='--',label=f"elegido={chosen['threshold']:.2f}");ax.legend();ax.set_xlabel('Threshold');ax.set_ylabel('Métrica');ax.set_title('RF tuned — threshold (validación 2024)');fig.tight_layout();fig.savefig(OUTF/'threshold_sweep.png',dpi=160);plt.close(fig)
fpr,tpr,_=roc_curve(yva,p);fig,ax=plt.subplots(figsize=(6,5));ax.plot(fpr,tpr,label=f"AUC={base['roc_auc']:.4f}");ax.plot([0,1],[0,1],'--');ax.legend();ax.set_title('ROC RF tuned');fig.tight_layout();fig.savefig(OUTF/'roc_rf_tuned.png',dpi=160);plt.close(fig)
pr,re,_=precision_recall_curve(yva,p);fig,ax=plt.subplots(figsize=(6,5));ax.plot(re,pr,label=f"AP={base['average_precision']:.4f}");ax.legend();ax.set_title('Precision-Recall RF tuned');fig.tight_layout();fig.savefig(OUTF/'pr_rf_tuned.png',dpi=160);plt.close(fig)
report={'scope':'SOLO 2024; 2025 NO leído','train_rows':int(len(ytr)),'validation_rows':int(len(yva)),'features':feats,'etnia':'excluida','winner_variant':'original_1_6|balanced','winner_params':{'n_estimators':160,'max_depth':12,'min_samples_leaf':10,'min_samples_split':2,'max_features':'sqrt','class_weight':'balanced_subsample','random_state':42},'fit_seconds':fit,'winner_threshold_0_5':base,'threshold_selected_on_validation_2024':chosen,'threshold_rule':'max F1 sujeto a recall>=0.80; desempate balanced_accuracy/precision','metas':{'roc_auc':.80,'balanced_accuracy':.75,'recall':.80,'f1':.75,'fnr_max':.20},'cumplimiento_threshold_elegido':{'roc_auc':chosen['roc_auc']>=.80,'balanced_accuracy':chosen['balanced_accuracy']>=.75,'recall':chosen['recall']>=.80,'f1':chosen['f1']>=.75,'fnr':chosen['fnr']<=.20},'tuning_note':'Exploración controlada 2024: profundidades 16/20/None empeoraron AUC frente a depth=12. Se probó class_weight=balanced_subsample; mejora recall/F1 a threshold 0.5. COD_DEPE original retiene mejor balance que agrupado en baseline.','versions':{'python':sys.version.split()[0],'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},'next_step':'Congelar parámetros y threshold, luego evaluar una sola vez sobre 2025 si el usuario aprueba.'}
with open(OUTR/'rf_tuning_2024.json','w',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
