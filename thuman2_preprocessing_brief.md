# Task: Preprocess THuman2.0 into CloSe-Di format and build a combined split

## Goal
Produce per-scan `.npz` files for THuman2.0 matching the CloSe-Di schema, then build a joint train/val/test split combining CloSe-Di + THuman2.0.

## Dependencies
- THuman2.0 scans: `.obj` + `material0.jpeg` per scan, one folder per scan ID (`0000`, `0001`, …) -> THuman2.0_Release_copy.zip -> unzip, move and rename folder to `CloSe-with-CBL-and-PTB/data/THuman2.0_Release_copy`
- THuman2.0 SMPL fits (Jinlong Yang / Xu Chen release): `.pkl` per scan with `betas`, `body_pose`, `global_orient`, `transl`, `scale`. **Not** SMPL-X. Found on https://github.com/ytrock/THuman2.0-Dataset readme. Put into `CloSe-with-CBL-and-PTB/data/THuman2.0_smpl`
- SMPL model files (`SMPL_NEUTRAL.pkl` etc.) at the path expected by `smplx.create(model_type='smpl')`.https://smplify.is.tue.mpg.de/login.php -> create account -> confirm account via email -> website -> downloads -> SMPLIFY_CODE_V2.zip -> Unzip -> smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl -> put into folder CloSe-with-CBL-and-PTB/models/smpl/SMPL_NEUTRAL.pkl (raname the file).

## prep_thuman.sbatch
Make sure the `BM_DIR_PATH` is set to the parent folder which contains `smpl/SMPL_NEUTRAL.pkl`

Run with `sbatch prep_thuman.sbatch` from main directory.

## Outputs
There should be a new folder called `./data/THuman2.0_processed` as well as a new config file `cfg/data_split_thuman.json` which you can then use for train test val splits. 