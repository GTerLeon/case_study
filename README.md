Implementation for functional dependency discovery inspired by Algorithm 3 IdentifyRedundancy in section 3 Bayesian network structure learning with determinism screening in the paper Bayesian networks for static and temporal data fusion by Thibaud Rahier.
https://theses.hal.science/tel-01971371v4/file/PhD_diffusion.pdf

Datasets:
iris.csv: for initial testing. exact 4FD 5 col, 150 row
adult.csv: test on larger size. exact 78 FD, 14 col, 48,842 row
ncvoter.csv: used for scalability test by pyro & aid-fd, good for approximate evaluation. partial 19 col, 64K-1M row 
uniprot.csv: 2nd best for partial. 30col, 250K row
