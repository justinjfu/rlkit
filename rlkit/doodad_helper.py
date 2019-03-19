import os.path as path
from doodad.easy_sweep import DoodadSweeper
import doodad.mount as mount

THIS_FILE_DIR = path.dirname(path.realpath(__file__))
SRC_DIR = path.dirname(THIS_FILE_DIR)
CODE_DIR = path.dirname(SRC_DIR)

MOUNTS = [
    mount.MountLocal(local_dir=SRC_DIR, pythonpath=True), # Code project folder
    #mount.MountLocal(local_dir=path.join(CODE_DIR, 'rlutil'), pythonpath=True), # RLLAB
]

SWEEPER_WEST1 = DoodadSweeper(mounts=MOUNTS,
                        python_cmd='python',
                        docker_img='justinfu/rlkit:0.1',
                        docker_output_dir='/data',
                        local_output_dir='data/docker_test_run',
                        gcp_bucket_name='justin-doodad',
                        gcp_image='docker-justinfu-dbq-0-1',
                        gcp_project='qlearning000',
)


SWEEPER_EAST1 = DoodadSweeper(mounts=MOUNTS,
                        python_cmd='python',
                        docker_img='justinfu/rlkit:0.1',
                        docker_output_dir='/data',
                        local_output_dir='data/docker_test_run',
                        gcp_bucket_name='justin-doodad-east1',
                        gcp_image='docker-justinfu-dbq-0-1',
                        gcp_project='qlearning000',
)

SWEEPER_CENTRAL1 = DoodadSweeper(mounts=MOUNTS,
                        python_cmd='python',
                        docker_img='justinfu/rlkit:0.1',
                        docker_output_dir='/data',
                        local_output_dir='data/docker_test_run',
                        gcp_bucket_name='justin-doodad-central1',
                        gcp_image='docker-justinfu-dbq-0-1',
                        gcp_project='qlearning000',
)
