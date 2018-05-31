Introduction: 
The need for the participants to collaborate with each other for this analysis gives rise to the concept of collaborative learning. However, the possibility of the cloud service being semi-honest poses a key challenge: preserving the participants’ privacy. We address this challenge with a two-stage scheme called RG+RP: in the first stage, each participant perturbs his/her data by passing the data through a nonlinear function called repeated Gompertz (RG); in the second stage, he/she then projects his/her data to a lower dimension in an (almost) distance-preserving manner, using a specific random projection (RP) matrix. The nonlinear RG function is designed to mitigate maximum a posteriori (MAP) estimation attacks, while random projection resists independent component analysis (ICA) attacks and ensures clustering accuracy. The proposed two-stage randomisation scheme is assessed in terms of its recovery resistance to MAP estimation attacks. Preliminary theoretical analysis as well as experimental results on synthetic and real-world datasets indicate that RG+RP has better recovery resistance to MAP estimation attacks than most state-of-the-art techniques. For clustering, fuzzy c-means (FCM) is used. Results using seven cluster validity indices (CVIs), root mean squared error (RMSE) and accuracy ratio show that clustering results based on two-stage-perturbed data are comparable to the clustering results based on raw data — this confirms the utility of our privacy-preserving scheme when used with either FCM or HCM.


How to run:
To reproduce the PrivacyTest result(repeated Gompertz+random projection matrix) for purely Gaussian datasets under maximum a priori (MAP) estimation attack, run below commands:
% 9=two_Gompertz+RP; 1=MAP estimation; 0=recovers normal points, 1=recover outliers
runPrivacyTest_DKE('Gaussian', 9, 1, 0);
runPrivacyTest_DKE('Gaussian', 9, 1, 1);
Concatenate and feed the RP or two-stage perturbed data to clustering algorithm to compute CVIs by running: 
fuzzy_twostage_s1
fuzzyComparisonCVI: compute senven CVIs, including ARI,RI,MI,NMIsqrt,VI,NVI,JVI

Requirements:
Matlab
KDE toolbox (https://www.ics.uci.edu/~ihler/code/kde.html):
Put @kde under the your directory and set path in matlab. For example:
/home/ihler/myMatlabCode/@kde, then add to my path: '/home/ihler/myMatlabCode'

Remember to cite the following papers if you use any of the code:
@inproceedings{lyu2017privacy,
  title={Privacy-Preserving Collaborative Deep Learning with Application to Human Activity Recognition},
  author={Lyu, Lingjuan and He, Xuanli and Law, Yee Wei and Palaniswami, Marimuthu},
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
  pages={1219--1228},
  year={2017},
  organization={ACM}
}
@article{lyu2018privacy,
  title={Privacy-preserving collaborative fuzzy clustering},
  author={Lyu, Lingjuan and Bezdek, James C and Law, Yee Wei and He, Xuanli and Palaniswami, Marimuthu},
  journal={Data \& Knowledge Engineering},
  year={2018},
  publisher={Elsevier}
}