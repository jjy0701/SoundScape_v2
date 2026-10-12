"""(04) 장면 단위 PyTorch Dataset + 거리/Ridge/작은 MLP baseline.
실행: python "train_baseline(04_기초모델학습).py"
필수: numpy, matplotlib, torch. 데이터/03번 분할이 같은 폴더에 있어야 합니다.
기본 CPU: 100개 소규모 실험의 설치/재현 편의를 위한 선택입니다.
선택 인자: --epochs 300 --patience 40 --seed 42 --device cpu
학습 입력 = 배치 6개 + receiver x,y + log10 거리 + log10 장면 최소거리.
최소거리는 기하정보만 사용. target 최대값을 입력으로 사용하지 않습니다.
train/validation만 로딩하며 test NPY는 읽지 않습니다.
출력은 results 아래 실행별 폴더. 평가 단위는 유효 receiver를 평균한 장면입니다.
"""
import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise SystemExit('PyTorch가 필요합니다. 현재 가상환경에서 python -m pip install torch 를 실행하세요.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GEOM = ['speaker_x','speaker_y','obstacle_x','obstacle_y','obstacle_w','obstacle_l']
FEATURES = GEOM+['receiver_x','receiver_y','log10_distance','log10_min_scene_distance']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def build_scene(dataset, sid, row, near, expected_hash):
    path=dataset/'npy'/f'{sid}.npy'
    if sha(path) != expected_hash:
        raise ValueError(f'{sid}: 03번 분할 이후 NPY가 변경됐습니다')
    X,Y=np.meshgrid(np.linspace(.5,float(row['room_x'])-.5,20),
                    np.linspace(.5,float(row['room_y'])-.5,20))
    sx,sy,ox,oy,w,h=[float(row[k]) for k in GEOM]
    invalid=(X>=ox)&(X<=ox+w)&(Y>=oy)&(Y<=oy+h)
    valid=(~invalid).reshape(-1)
    a=np.load(path,allow_pickle=False)
    if a.shape != (20,20) or not np.array_equal(np.isnan(a),invalid) or not np.isfinite(a[~invalid]).all():
        raise ValueError(f'{sid}: shape/NaN mask/유한값 오류; 02번 검사 필요')
    if not valid.any() or abs(float(a[~invalid].max()))>1e-5:
        raise ValueError(f'{sid}: 유효 영역 또는 최대값 정규화 오류')
    distance=np.hypot(X-sx,Y-sy).reshape(-1)
    dmin=float(distance[valid].min())
    if dmin<=0 or bool(dmin<.1)!=near:
        raise ValueError(f'{sid}: 최소거리/근접 표시 불일치')
    f=np.column_stack([np.full(400,float(row[k])) for k in GEOM]+
                      [X.ravel(),Y.ravel(),np.log10(np.maximum(distance,1e-8)),
                       np.full(400,np.log10(dmin))]).astype(np.float32)
    # 무효 target만 0으로 채워 NaN 연산을 막습니다. loss에서 제외됩니다.
    target=np.where(invalid,0,a).reshape(-1).astype(np.float32)
    # 비교용 역거리 모델: 반사/차폐 미포함. 최대값은 거리에서 직접 계산합니다.
    distance_prediction=(-20*np.log10(np.maximum(distance,1e-8)/dmin)).astype(np.float32)
    return dict(id=sid, x=f, y=target, mask=valid, near=near, X=X,Y=Y,
                geom=[sx,sy,ox,oy,w,h], distance_prediction=distance_prediction)


class AcousticDataset(Dataset):
    """한 item이 장면 1개: x(400,10), target(400), valid_mask(400), ID."""
    def __init__(self, scenes, x_mean, x_std, y_mean, y_std):
        self.scenes=scenes
        self.x=torch.from_numpy(np.stack([(s['x']-x_mean)/x_std for s in scenes]).astype(np.float32))
        self.y=torch.from_numpy(np.stack([(s['y']-y_mean)/y_std for s in scenes]).astype(np.float32))
        self.mask=torch.from_numpy(np.stack([s['mask'] for s in scenes]))
    def __len__(self): return len(self.scenes)
    def __getitem__(self,i):
        return self.x[i],self.y[i],self.mask[i],self.scenes[i]['id']


def masked_mse(pred,target,mask):
    # 각 장면에 동일 가중치: 장애물 크기에 따른 유효 receiver 수 차이를 보정합니다.
    err=(pred-target).square().masked_fill(~mask,0)
    return (err.sum(dim=1)/mask.sum(dim=1)).mean()


def measure(scenes,pred):
    if not np.isfinite(pred).all(): raise ValueError('예측에 NaN/무한대 발생')
    rows=[]
    for s,p in zip(scenes,pred):
        e=p[s['mask']]-s['y'][s['mask']]
        rows.append(dict(sample_id=s['id'],near_source=s['near'],
                         mae_db=float(np.abs(e).mean()),rmse_db=float(np.sqrt(np.mean(e**2)))))
    groups={}
    for name,rs in [('all',rows),('near',[r for r in rows if r['near_source']]),
                    ('non_near',[r for r in rows if not r['near_source']])]:
        groups[name]=dict(n=len(rs),mae_db=float(np.mean([r['mae_db'] for r in rs])) if rs else None,
                         mean_scene_rmse_db=float(np.mean([r['rmse_db'] for r in rs])) if rs else None)
    return groups,rows


def fit_ridge(train, val, xm, xs, ym, ys):
    # 평균 장면 MSE + alpha*||w||², 절편에는 페널티 없음.
    A=np.concatenate([(s['x'][s['mask']]-xm)/xs for s in train]).astype(np.float64)
    A=np.column_stack([A,np.ones(len(A))])
    b=np.concatenate([(s['y'][s['mask']]-ym)/ys for s in train]).astype(np.float64)
    weights=np.concatenate([np.full(s['mask'].sum(),1/(len(train)*s['mask'].sum())) for s in train])
    gram=A.T@(weights[:,None]*A); rhs=A.T@(weights*b)
    penalty=np.eye(A.shape[1]); penalty[-1,-1]=0
    V=np.stack([np.column_stack([(s['x']-xm)/xs,np.ones(400)]) for s in val])
    candidates=[]; best=None
    for alpha in [1e-6,1e-4,1e-2,1.0]:
        coeff=np.linalg.solve(gram+alpha*penalty,rhs)
        p=(V@coeff)*ys+ym
        score=measure(val,p)[0]['all']['mean_scene_rmse_db']
        candidates.append(dict(alpha=alpha,validation_mean_scene_rmse_db=score))
        if best is None or score<best[0]: best=(score,alpha,coeff,p)
    return best,candidates


def plot_comparisons(out, scenes, predictions, train):
    # validation 정렬 후 처음 3개: 오차가 작은 그림만 고르지 않습니다.
    low=min(-1.0,float(np.percentile(np.concatenate([s['y'][s['mask']] for s in train]),1)))
    selected=scenes[:3]
    error_max=max(1.,max(float(np.max(np.abs(predictions['mlp'][i][s['mask']]-s['y'][s['mask']]))) for i,s in enumerate(selected)))
    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('white')
    for i,s in enumerate(selected):
        fig,axes=plt.subplots(1,5,figsize=(19,5),layout='constrained')
        arrays=[s['y'],predictions['distance'][i],predictions['ridge'][i],predictions['mlp'][i],predictions['mlp'][i]-s['y']]
        for j,(ax,a,title) in enumerate(zip(axes,arrays,['Simulation','Distance','Ridge','MLP','MLP - simulation'])):
            data=np.where(s['mask'],a,np.nan).reshape(20,20)
            im=ax.pcolormesh(s['X'],s['Y'],data,shading='auto',cmap='coolwarm' if j==4 else cmap,
                             vmin=-error_max if j==4 else low,vmax=error_max if j==4 else 0)
            sx,sy,ox,oy,w,h=s['geom']
            ax.add_patch(plt.Rectangle((ox,oy),w,h,facecolor='white',edgecolor='black'))
            ax.scatter(sx,sy,marker='*',color='red',s=70)
            ax.set(title=title,xlabel='x (m)',ylabel='y (m)',aspect='equal')
            fig.colorbar(im,ax=ax,label='dB',shrink=.7,extend='both')
        fig.suptitle(s['id']+' | validation | shared level scale; colors may saturate')
        fig.savefig(out/f"{s['id']}(예측비교).png",dpi=130)
        plt.close(fig)
    return dict(level_vmin=low,level_vmax=0,error_abs_max=error_max,
                selection='first 3 validation IDs; raw predictions not clipped/renormalized')


def main():
    project_root=next(parent for parent in Path(__file__).resolve().parents
                      if (parent / "configs" / "classroom").is_dir()
                      and (parent / "data").is_dir()
                      and (parent / "results").is_dir())
    base=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=project_root/'data'/'surrogate_2d'/'dataset')
    parser.add_argument('--output-root',type=Path,default=project_root/'results'/'surrogate_2d')
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--patience',type=int,default=40)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.epochs<1 or args.patience<1: parser.error('epochs/patience는 양수여야 합니다')
    if args.device=='cuda' and not torch.cuda.is_available(): raise ValueError('이 Python 환경에서 CUDA를 사용할 수 없습니다. --device cpu 로 실행하세요')
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    dataset=args.dataset.resolve()
    mp=dataset/'splits/split_manifest.json'
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    if sha(dataset/'metadata.csv')!=manifest['metadata_sha256']: raise ValueError('03번 분할 이후 metadata 변경 감지')
    ids=manifest['splits']; all_ids=sum([ids[k] for k in ['train','validation','test']],[])
    if [len(ids[k]) for k in ['train','validation','test']]!=[70,15,15] or len(set(all_ids))!=100:
        raise ValueError('분할 수량/중복 오류')
    for k in ids:
        if (dataset/'splits'/f'{k}_ids.txt').read_text().splitlines()!=ids[k]: raise ValueError('분할 ID 목록 불일치')
    with (dataset/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:
        rows={f"sample_{int(r['sample_id'].removeprefix('sample_')):04d}":r for r in csv.DictReader(f)}
    near={r['sample_id']:bool(r['near_source']) for r in manifest['samples']}
    data={k:[build_scene(dataset,sid,rows[sid],near[sid],manifest['npy_sha256'][sid]) for sid in ids[k]] for k in ['train','validation']}
    train,val=data['train'],data['validation']
    tx=np.concatenate([s['x'][s['mask']] for s in train]); ty=np.concatenate([s['y'][s['mask']] for s in train])
    xm=tx.mean(axis=0,dtype=np.float64).astype(np.float32); xs=tx.std(axis=0,dtype=np.float64).astype(np.float32); xs[xs<1e-6]=1
    ym=float(ty.mean(dtype=np.float64)); ys=max(float(ty.std(dtype=np.float64)),1e-6)
    train_ds=AcousticDataset(train,xm,xs,ym,ys); val_ds=AcousticDataset(val,xm,xs,ym,ys)
    loader=DataLoader(train_ds,batch_size=10,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(args.seed))
    model=nn.Sequential(nn.Linear(len(FEATURES),64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1)).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    vx=val_ds.x.to(args.device)
    out=args.output_root.resolve()/('baseline(04_기초모델학습)_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    config=dict(seed=args.seed,device=args.device,torch_version=str(torch.__version__),numpy_version=np.__version__,
                epochs=args.epochs,patience=args.patience,features=FEATURES,hidden_sizes=[64,64],
                batch_size=10,learning_rate=.001,split_manifest_sha256=sha(mp),script_sha256=sha(Path(__file__)),
                x_mean=xm.tolist(),x_std=xs.tolist(),y_mean=ym,y_std=ys,
                metric='arithmetic mean of per-scene masked RMSE (dB)',
                postprocessing='none: no clipping or per-prediction normalization',
                test_targets_loaded=False,limits='pilot: 100 scenes, fixed room/material; ISM order convergence not established')
    write_json(out/'config.json',config)
    predictions={'distance':np.stack([s['distance_prediction'] for s in val])}
    ridge,candidates=fit_ridge(train,val,xm,xs,ym,ys)
    predictions['ridge']=ridge[3]
    np.savez(out/'ridge.npz',coefficients=ridge[2],alpha=ridge[1],x_mean=xm,x_std=xs,y_mean=ym,y_std=ys)
    print('(04) 기초모델학습 v1',flush=True)
    print(f'train=70 / validation=15 / device={args.device} / test 평가 안 함',flush=True)
    for name,p in predictions.items():
        print(f"{name}: validation mean-scene RMSE={measure(val,p)[0]['all']['mean_scene_rmse_db']:.4f} dB",flush=True)
    best=float('inf'); state=None; best_epoch=0; stale=0; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=0.; seen=0
        for x,y,mask,_ in loader:
            x,y,mask=x.to(args.device),y.to(args.device),mask.to(args.device)
            optimizer.zero_grad(); loss=masked_mse(model(x).squeeze(-1),y,mask)
            if not torch.isfinite(loss): raise ValueError('학습 loss NaN/무한대')
            loss.backward(); optimizer.step(); loss_sum+=float(loss.detach().cpu())*len(x); seen+=len(x)
        model.eval()
        with torch.no_grad(): vp=model(vx).squeeze(-1).cpu().numpy()*ys+ym
        score=measure(val,vp)[0]['all']['mean_scene_rmse_db']
        history.append(dict(epoch=epoch,train_masked_mse_db2=loss_sum/seen*ys**2,validation_mean_scene_rmse_db=score))
        if score<best:
            best=score; best_epoch=epoch; stale=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(model_state_dict=state,config=config,best_epoch=epoch,validation_rmse_db=best),out/'mlp_best.pt')
        else: stale+=1
        if epoch==1 or epoch%10==0: print(f'epoch {epoch:03d} | train MSE={history[-1]["train_masked_mse_db2"]:.4f} dB² | val RMSE={score:.4f} dB | best={best:.4f}',flush=True)
        if stale>=args.patience:
            print(f'조기 종료: {args.patience} epoch 동안 validation 개선 없음',flush=True); break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        predictions['mlp']=model(vx).squeeze(-1).cpu().numpy()*ys+ym
        tp=model(train_ds.x.to(args.device)).squeeze(-1).cpu().numpy()*ys+ym
    report=dict(best_epoch=best_epoch,train_mlp=measure(train,tp)[0],validation={},ridge_candidates=candidates,test_evaluated=False)
    detail=[]
    for name,p in predictions.items():
        report['validation'][name],rs=measure(val,p)
        detail.extend([dict(model=name,**r) for r in rs])
        saved=np.stack([np.where(s['mask'],pp,np.nan).reshape(20,20) for s,pp in zip(val,p)])
        np.savez(out/f'validation_{name}.npz',sample_ids=np.array(ids['validation']),prediction=saved)
    for filename,rs in [('history(01_학습기록).csv',history),('validation(02_샘플별오차).csv',detail)]:
        with (out/filename).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    report['plot_scale']=plot_comparisons(out,val,predictions,train)
    write_json(out/'metrics(03_평가요약).json',report)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot([h['epoch'] for h in history],[h['validation_mean_scene_rmse_db'] for h in history]);ax.set(xlabel='Epoch',ylabel='Validation mean scene RMSE (dB)');ax.axvline(best_epoch,color='gray',ls='--');fig.tight_layout();fig.savefig(out/'learning_curve(04_학습곡선).png',dpi=140);plt.close(fig)
    print(f'\n학습 완료 | best epoch={best_epoch}',flush=True)
    for name,groups in report['validation'].items():
        g=groups['all'];print(f"{name}: MAE={g['mae_db']:.4f} dB / mean-scene RMSE={g['mean_scene_rmse_db']:.4f} dB")
    print('근접/비근접 상세 오차: metrics(03_평가요약).json')
    print(f'결과 폴더: {out}')
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,OSError,KeyError) as exc:
        print(f'FAIL: {exc}');raise SystemExit(1)
