import numpy as np
from scipy import stats
import matplotlib as mpl
from . import constants as C
from scipy.stats import rankdata
mpl.use('Agg')

############################
##    Ths Statistics      ##
############################

def Get_Ths(pts_corr, seg, ins, ths, ths_, cnt):

    pts_in_ins = {}
    for ip, pt in enumerate(pts_corr):
        if ins[ip] in pts_in_ins.keys():
            pts_in_curins_ind = pts_in_ins[ins[ip]]
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip])
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)

            if seg[ip]==8:
                print(bin)

            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)

            if numpt_notin_curins > 0:

                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):
                    if b == 0:
                        break
                    tp = float(np.sum(pt[pts_in_curins_ind] < bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind] < bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp/fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[seg[ip]] += bin[ib_opt]
                    ths_[seg[ip]] += bin[ib_opt]
                    cnt[seg[ip]] += 1

        else:
            pts_in_curins_ind = (ins == ins[ip])
            pts_in_ins[ins[ip]] = pts_in_curins_ind
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip])
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)

            if seg[ip]==8:
                print(bin)

            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)

            if numpt_notin_curins > 0:

                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):

                    if b == 0:
                        break

                    tp = float(np.sum(pt[pts_in_curins_ind]<bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind]<bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp / fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[seg[ip]] += bin[ib_opt]
                    ths_[seg[ip]] += bin[ib_opt]
                    cnt[seg[ip]] += 1

    return ths, ths_, cnt


##############################
##    Merging Algorithms    ##
##############################

def GroupMerging(pts_corr, confidence, seg, label_bin,
    conf_thresh=C.DEFAULT_CONFIDENCE_THRESHOLD):

    # Filters points based on confidence level
    confvalidpts = (confidence>conf_thresh)

    # Number of unique segementation classes
    un_seg = np.unique(seg)
    refineseg = -1* np.ones(pts_corr.shape[0])
    groupid = -1* np.ones(pts_corr.shape[0])
    numgroups = 0
    groupseg = {}

    # Iterate over each segmentation class
    for i_seg in un_seg:
        if i_seg==-1:
            continue
        pts_in_seg = (seg==i_seg)
        valid_seg_group = np.where(pts_in_seg & confvalidpts)
        proposals = []
        if valid_seg_group[0].shape[0]==0:
            proposals += [pts_in_seg]
        else:
            for ip in valid_seg_group[0]:
                validpt = (pts_corr[ip] < label_bin[i_seg]) & pts_in_seg
                if np.sum(validpt)>5:
                    flag = False
                    for gp in range(len(proposals)):
                        iou = float(np.sum(validpt & proposals[gp])) / np.sum(validpt|proposals[gp])#uniou
                        validpt_in_gp = float(np.sum(validpt & proposals[gp])) / np.sum(validpt)#uniou
                        if iou > 0.6 or validpt_in_gp > 0.8:
                            flag = True
                            if np.sum(validpt)>np.sum(proposals[gp]):
                                proposals[gp] = validpt
                            continue

                    if not flag:
                        proposals += [validpt]

            if len(proposals) == 0:
                proposals += [pts_in_seg]

        # print("Group Merging has found {} potential proposals for Segmentation class {}".format(len(proposals), i_seg))
        for gp in range(len(proposals)):
            # print("Group Proposal {} for class {} has {} points".format(gp, i_seg ,np.sum(proposals[gp])))
            if np.sum(proposals[gp])>C.MIN_POINTS_IN_GROUP_PROPOSAL:
                groupid[proposals[gp]] = numgroups
                groupseg[numgroups] = i_seg
                numgroups += 1
                refineseg[proposals[gp]] = stats.mode(seg[proposals[gp]])[0]

    #return groupid, refineseg, groupseg

    '''
    un, cnt = np.unique(groupid, return_counts=True)
    for ig, g in enumerate(un):
        if cnt[ig] < MIN_POINTS_IN_GROUP_PROPOSAL:
            groupid[groupid==g] = -1
    '''

    un, cnt = np.unique(groupid, return_counts=True)
    groupidnew = groupid.copy()
    for ig, g in enumerate(un):
        if g == -1:
            continue
        groupidnew[groupid==g] = (ig-1)
        groupseg[(ig-1)] = groupseg.pop(g)
    groupid = groupidnew

    for ip, gid in enumerate(groupid):
        if gid == -1:
            pts_in_gp_ind = (pts_corr[ip] < label_bin[seg[ip]])
            pts_in_gp = groupid[pts_in_gp_ind]
            pts_in_gp_valid = pts_in_gp[pts_in_gp!=-1]
            if len(pts_in_gp_valid) != 0:
                groupid[ip] = stats.mode(pts_in_gp_valid)[0][0]

    return groupid, refineseg, groupseg

# Takes a list of labels and relabels them to [0-#labels] based on order
# For example, [-1, 3, 4] becomes [0,1,2]
def obtain_rank(data):
    return (rankdata(data, method='dense') - 1)
