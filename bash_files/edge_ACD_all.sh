

bash bash_files/edge_ACD.sh seas_gpu _hc cf gt hc_clip 1e-3 3e-3 1e-4 5e-4 2e-5 3e-5 5e-5;
bash bash_files/edge_ACD.sh seas_gpu _hc oa gt hc_clip 1e-3 5e-3 1e-4 1.5e-4 3e-4 3e-5 5e-5;
bash bash_files/edge_ACD.sh seas_gpu _hc mean gt hc_clip 1e-3 1e-4 2e-4 5e-4 3e-5 5e-5 7e-5;
bash bash_files/edge_ACD.sh seas_gpu _hc resample gt hc_clip 1e-3 2e-3 1e-4 2e-4 5e-4 5e-5 7e-5;
bash bash_files/edge_ACD.sh seas_gpu _hc cf ioi hc_clip 1e-3 1.5e-3 3e-3 5e-3 1e-4 1.5e-4 2e-4 5e-4 7e-5;
bash bash_files/edge_ACD.sh seas_gpu _hc oa ioi hc_clip 1e-3 2e-3 1.2e-4 1.5e-4 2e-4 5e-4;
bash bash_files/edge_ACD.sh seas_gpu _hc mean ioi hc_clip 1e-3 2e-3 5e-3 2e-4 3e-4 4e-4 5e-4;
bash bash_files/edge_ACD.sh seas_gpu _hc resample ioi hc_clip 1e-3 1.5e-3 2e-3 5e-3 5e-4 6e-4 7.5e-4