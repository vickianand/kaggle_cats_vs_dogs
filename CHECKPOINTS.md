## Commit hash for the checkpoints

* M1: 824a4e6dea7830c45627b33ff8206214a45e4409
* M2: e512aeefd7a2c92b58a9ebdb4db28e70be3771c7
* M3: b427ecc99061eef71729cbef3385914d95dea64f
* M4: 4bf663731e42565a9046041afffc664c22aeb0ef
* M5: 3e95cdbe48133f8fd4a0ee106fb6201536f419dd
* M6: 3e95cdbe48133f8fd4a0ee106fb6201536f419dd

## Training M5 and M6
Note that M5 and M6 have same checkpoint. M6 has same settings as M5, only that M6 had decreasing learning-rate.

* Train M5:

```
python train.py --batch_norm --train_folder <path to folder having images>
```

* Train M6:

```
python train.py --batch_norm --decreasing_lr --train_folder <path to folder having images>
```
