import numpy as np
def get_key(target,mask, i):
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
    #if i <= 52 or target[mask].sum()/255 > int(mask.sum()*2/5):
        # print("{} ----- {}".format(target_upper,(upper+lower)/2))
    #    if ((upper+lower)/2 > target_upper):
    #        return True
    if i <= 52:
        if target_upper - upper < (lower - upper) / 4:
            return True
    elif target_upper - upper < (lower - upper) / 5 or target[mask].sum()/255 > int(mask.sum()*1/5):
        return True
    return False