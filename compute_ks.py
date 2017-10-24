#coding:utf-8
def compute_ks(data, group_num):
    sorted_list = data.sort_values(['predict'], ascending=True)
    group_scale = sorted_list.shape[0] / group_num
    total_good = sorted_list['label'].sum()
    total_bad = sorted_list.shape[0] - total_good
    max_ks = 0.0
    sum_good = 0
    sum_bad = 0
    for i in range(0, group_num):
        #考虑数据总量不能整除组数时，最后一组的情况
        if i == group_num - 1:
            for j in range(i * group_scale, sorted_list.shape[0]):
                if sorted_list['label'][j] == 1:
                    sum_good += 1
                else:
                    sum_bad += 1
                temp_ks = abs(sum_good / total_good - sum_bad / total_bad)
                max_ks = max(temp_ks, max_ks)
        else:
            for j in range(i * group_scale, (i + 1) * group_scale):
                if sorted_list['label'][j] == 1:
                    sum_good += 1
                else:
                    sum_bad += 1
                temp_ks = abs(sum_good / total_good - sum_bad / total_bad)
                max_ks = max(temp_ks, max_ks)
    return  max_ks