Dependencies:
===

Python >= 3.7

PyTorch = 1.2.0

numpy = 1.22.3 

pandas = 1.2.1

scikit-learn = 1.0.2

scipit = 1.8.0

Datasets:
===
Junyi:
---
https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1275

NIPS:
---
https://eedi.com/projects/neurips-education-challenge

ASSIST: 
---
https://sites.google.com/site/assistmentsdata



References:
===
AKT: https://github.com/arghosh/AKT

SAINT+: https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing-

DIMKT: https://github.com/shshen-closer/DIMKT

SAKT: https://github.com/arshadshk/SAKT-pytorch

PKT: https://github.com/WeiMengqi934/PKT

spareseKT: https://github.com/pykt-team/pykt-toolkit

DKT: https://github.com/mmkhajah/dkt

DKVMN: https://github.com/jennyzhang0215/DKVMN

LBKT: https://github.com/bigdata-ustc/EduKTM/tree/main/EduKTM/LBKT

LPKT: https://github.com/bigdata-ustc/EduKTM/tree/main/EduKTM/LPKT

EduKTM: https://github.com/bigdata-ustc/EduKTM

Train and Test
===

Please click on the link above to download the corresponding datasets. Then, change the location where the datasets are stored in the run.py, and use this file to train the model. We also provide the comparison methods in Baselines. To run these methods, empoly the prepare_data.py in the ClusterKT folder to read data according to the model's input. In the Baselines config.py file, we place the code for exercise difficulty and concept difficulty required in DIMKT, and for Q matrix construction required in LBKT and LPKT.

