import numpy as np
def get_key(target,mask):
    height = mask.shape[0]
    upper = 0
    lower = height - 1
    while (upper < height) and (mask[upper,:].sum()==0):
        upper = upper + 1
    while (lower > upper) and (mask[lower,:].sum()==0):
        lower = lower - 1
    target_with_mask = target * mask
    target_upper = 0
    target_lower = height - 1
    while (target_upper < height) and (target_with_mask[target_upper,:].sum()==0):
        target_upper = target_upper + 1
    # while (target_lower > upper) and (target_with_mask[target_lower,:].sum()==0):
    #     target_lower = target_lower - 1
    # import pdb;pdb.set_trace()
    if target[mask].sum()/255 > int(mask.sum()*2/3):
        # print("{} ----- {}".format(target_upper,(upper+lower)/2))
        if ((upper+lower)/2 > target_upper):
            return True
    return False