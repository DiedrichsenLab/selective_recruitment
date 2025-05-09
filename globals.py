from pathlib import Path

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/Users/lshahsha/Documents/data/FunctionalFusion'

conn_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/connectivity'
if not Path(conn_dir).exists():
    conn_dir = '/srv/diedrichsen/data/Cerebellum/connectivity'
if not Path(conn_dir).exists():
    conn_dir = '/Users/lshahsha/Documents/data/Cerebellum/connectivity'

atlas_dir = base_dir + '/Atlases'
