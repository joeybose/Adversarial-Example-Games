# Defenses

This contains sample commands to run all of our Baseline models, taken from the
original git repos but repurposed for the NoBox codebase.


# Ensemble Adversarial Training (Kurakin et. al 2018)
```
python ensemble_adver_train_mnist.py --model modelD_ens --adv_models modelA modelC modelB --type=3 --epochs=12 --train_adv
```



## License
[MIT](https://choosealicense.com/licenses/mit/)


