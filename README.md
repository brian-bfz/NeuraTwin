## Install
Download PhysTwin/data, PhysTwin/experiments, PhysTwin/experiments_optimization, PhysTwin/gaussian_output

```bash scripts/env_install.sh```

```conda activate NeuraTwin```

If error pops up for urdfpy, click on the problematic file, and change ```np.float``` to ```float```

## PhysTwin
```python -m PhysTwin.interactive_pte --case_name single_push_rope --gnn_model```

```python -m PhysTwin.generate_data --case_name single_push_rope --episodes```

```python -m PhysTwin.v_from_d --case_name single_push_rope --episodes```

```python -m PhysTwin.html_video```

```python -m PhysTwin.control --case_name single_push_rope --dir_name --episode```

## GNN
```python -m GNN.dataset.preprocess --data_file```

```python -m GNN.train.train_gnn_dyn --name```

```CUDA_VISIBLE_DEVICES=0 python -m GNN.train.train_gnn_dyn```

```python -m GNN.inference --model --episodes --video```

```python -m GNN.html_video --model```

```python -m GNN.plot_training_loss --model --save_plots```

```python -m GNN.control --model --dir_name --episode```
