import numpy as np
import IOU
import DeltaTheta
import KF
import Track
from scipy.optimize import linear_sum_assignment


consistency_lambda = 0.8
#C(X,Zeta) = Ciou(X,Zeta) + lambdaCconsistency(Z,Zeta)
tracks = [[0.617,0.3594420600858369,0.114,0.17381974248927037]] #detections from frame 1
observations = [[0.094,0.38626609442060084,0.156,0.23605150214592274]] #detections from frame 2
Cost_matrix = []
for i in tracks:
    Cost_line = []
    for j in observations:
        Cost = IOU.IOU(i,j) + consistency_lambda*DeltaTheta.DeltaTheta(i.get_past(),j,i) ###Consider temporarily removing DeltaTheta to avoid confusion
        Cost_line = np.append(Cost_line,Cost)
    Cost_matrix = np.append(Cost_matrix,Cost_line)
row_ind, col_ind = linear_sum_assignment(Cost_matrix)

###
if len(tracks)<len(observations): #Determine wether to assign by row or column
    for i in range(len(col_ind)): 
        if (Track.check_filter(tracks[i])):
            tracks[i].update() #TODO: update with unmatched_observations[col_ind[i]]
            #TODO move matched track to matched
        else:
            tracks[i].init_filter(observations[col_ind[i]])
            #TODO move matched track to matched
else:
    for i in range(len(row_ind)):
        if ((Track.check_filter(tracks[row_ind[i]]))):
            tracks[row_ind[i]].update() #TODO: update with unmatched_observations[i]
                        #TODO move matched track to matched
        else:
            tracks[row_ind[i]].init_filter(observations[col_ind[i]])
                        #TODO move matched track to matched

###
#print(Cost_matrix)
#np.arctan()


#OCR for list tracks
#OCR
unmatched_tracks = [[0.617,0.3594420600858369,0.114,0.17381974248927037]]
unmatched_observations = [[0.094,0.38626609442060084,0.156,0.23605150214592274]] #observations
Cost_matrix = []
for i in unmatched_tracks:
    Cost_line = []
    for j in unmatched_observations:
        if (KF.check_filter(i)): #check if tracks need filter or not
            Cost = IOU.IOU(i.get_prediction(),j)
        else:
            Cost = IOU.IOU(i.get_past(),j)
        Cost_line = np.append(Cost_line,Cost)
    Cost_matrix = np.append(Cost_matrix,Cost_line)
row_ind, col_ind = linear_sum_assignment(Cost_matrix)
if len(unmatched_tracks)<len(unmatched_observations): #Determine wether to assign by row or column
    for i in range(len(col_ind)): 
        if (Track.check_filter(unmatched_tracks[i])):
            unmatched_tracks[i].update() #TODO: update with unmatched_observations[col_ind[i]]
            #TODO move matched track to matched
        else:
            unmatched_tracks[i].init_filter(unmatched_observations[col_ind[i]])
            #TODO move matched track to matched
else:
    for i in range(len(row_ind)):
        if ((Track.check_filter(unmatched_tracks[row_ind[i]]))):
            unmatched_tracks[row_ind[i]].update() #TODO: update with unmatched_observations[i]
                        #TODO move matched track to matched

        else:
            unmatched_tracks[row_ind[i]].init_filter(unmatched_observations[col_ind[i]])
                        #TODO move matched track to matched




print(Cost_matrix)
