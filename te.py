import utils
path1 = "/home/yangruixiong/ASL2/ASL10_data/concat-VCTK-singing-2s-8k/test_data_x_mfcc_lfcc0.npy"
path2 = "/home/yangruixiong/ASL2/ASL10_data/concat-VCTK-singing-2s-8k/test_data_y_mfcc_lfcc0.npy"
utils.npy_to_npz(path1,path2)
