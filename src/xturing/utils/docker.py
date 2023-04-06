import os
import subprocess


def run_docker_container(image, port_mapping, env_vars=None, gpus=None):
    cmd = ["docker", "container", "run"]

    if gpus is not None:
        cmd.extend(["--gpus", gpus])

    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])

    cmd.extend(["-p", port_mapping, image])

    subprocess.run(cmd)
