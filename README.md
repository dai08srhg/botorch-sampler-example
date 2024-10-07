# botorch-sampler-candidates-func
 OptunaのBoTorchSamplerのためのcandidates_funcを実装
 
 [`optuna.integration.BoTorchSampler`](https://optuna.readthedocs.io/en/v3.0.0/reference/generated/optuna.integration.BoTorchSampler.html)
 ```Python
 class optuna.integration.BoTorchSampler(*, candidates_func=None, constraints_func=None, n_startup_trials=10, independent_sampler=None, seed=None)
 ```
 > - **candidates_func** (Optional[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]]) –  
An optional function that suggests the next candidates. It must take the training data, the objectives, the constraints, the search space bounds and return the next candidates. The arguments are of type torch.Tensor. The return value must be a torch.Tensor. However, if constraints_func is omitted, constraints will be None. For any constraints that failed to compute, the tensor will contain NaN.  
If omitted, it is determined automatically based on the number of objectives. If the number of objectives is one, Quasi MC-based batch Expected Improvement (qEI) is used. If the number of objectives is either two or three, Quasi MC-based batch Expected Hypervolume Improvement (qEHVI) is used. Otherwise, for larger number of objectives, the faster Quasi MC-based extended ParEGO (qParEGO) is used.


# candidates_func
以下の`candidates_func`を実装
- Single Objective
    - qEI (with Gamma Prior)
    - qLogEI (with Gamma Prior)
    - qEI
    - qLogEI
    - LCB
    - SAAS + EI
    - ~~Thompson Sampling~~
- Multi Objective
    - ~~qEHVI~~
    - ~~qLogEHVI~~

# Remarks
- Use BoTorch==0.12.0
    - 高次元BOの手法として，VanilaBO[[Carl Hvarfner, et al., 2024](https://arxiv.org/abs/2402.02229)]が提案されているが，BoTorchではv0.12.0からデフォルトになっている．([release v0.12.0](https://github.com/pytorch/botorch/releases/tag/v0.12.0))
    - "with Gamma Prior"でないものは，デフォルトのLogNormal Priorを利用
- 指定しない場合はqEIもしくはqEHVIが利用されると記載されているが，実際にはqLogEI, qLogEHVIが利用されている模様．
- SAASBOはfittingにNUTS Samplerを利用しており，計算量がO(N^3)なので実行速度が遅い．

# Sample
Build image
```bash
$ docker compose up --build
```
Run experiment
```
$ python main.py 

Run experiment: StyblinskiTang40
Start trial:1
Start optimization using TPE
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:06<00:00, 14.98it/s]
```

