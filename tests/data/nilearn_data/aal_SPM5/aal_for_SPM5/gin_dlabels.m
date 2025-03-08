function [] = gin_dlabels()
% ______________________________________________________________________
%
% Local Maxima Labeling : gin_dlabels.m
% Functional Maps are thresholded and the clusters and local maxima are
% extracted.
% The function is the assignment of the label of the 3D volume of interest
% it belongs to. The three nearest anatomical regions are listed.
%
% gin_dlabels.m		B Landeau - 10/09/03 - aal for SPM2
% ______________________________________________________________________


% get the SPM.mat file.
%[SPM,VOL,xX,xCon,xSDM] = spm_getSPM;
[SPM,xSPM] = spm_getSPM;

% Compute and Display Labels (and distances) for local max and for cluster
gin_list_dlabels('List',xSPM);
