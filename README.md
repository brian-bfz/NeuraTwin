## Install
Download PhysTwin/data, PhysTwin/experiments, PhysTwin/experiments_optimization, PhysTwin/gaussian_output

```bash scripts/env_install.sh```

```conda activate NeuraTwin```

## PhysTwin
```python -m PhysTwin.interactive_pte --case_name single_push_rope```

When error pops up, click on the problematic file, and change ```np.float``` to ```float```

```python -m PhysTwin.generate_data --case_name single_push_rope --episodes```

```python -m PhysTwin.v_from_d --case_name single_push_rope --episodes```

```python -m PhysTwin.html_video```

## GNN
```python -m GNN.train.train_gnn_dyn --name```

```python -m GNN.inference --model --episodes --video```

```python -m GNN.html_video --model```

```python -m GNN.plot_training_loss --model --save_plots```
