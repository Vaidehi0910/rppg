import os

tmp = "C:/rppg/tmp_physbench"
if not os.path.exists(tmp):
    os.makedirs(tmp)

dataset_ccnu = "C:"
dataset_mmpd = "C:/MMPD"
dataset_pure = "C:/PURE"
dataset_ubfc_rppg2 = "C:/UBFC-rPPG/DATASET_2"
dataset_ubfc_phys = "C:/UBFC-PHYS"
dataset_scamps = 'C:/scamps_videos'
dataset_cohface = 'C:/cohface/data'

# Please first generate these data through dataset_process.ipynb.
test_set_CCNU = "C:/rppg/ccnu_dataset_test.h5"
test_set_CCNU_rPPG = "C:/rppg/ccnu_rppg_dataset_test.h5"
test_set_PURE = "C:/rppg/pure_dataset.h5"
test_set_UBFC_rPPG2 = "C:/rppg/ubfc_rppg2_dataset.h5"
test_set_UBFC_PHYS = "C:/rppg/ubfc_phys_dataset.h5"
test_set_MMPD = "C:/rppg/mmpd_dataset.h5"
test_set_COHFACE = "C:/rppg/cohface_dataset.h5"
test_set_COHFACE_gray="C:/rppg/cohface_dataset_gray.h5"
# test_set_COHFACE_rPPG ="C:/rppg/cohface"