import os
import subprocess


def get_mlflow_model_path():
    current_file_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    return os.path.join(project_path, "model_result", "mlflow_model")


def execute_command(command):
    subprocess.run(command, shell=True, check=True)


commands_to_execute = [
    f"mlflow models serve -m {get_mlflow_model_path()} --env-manager local --host 127.0.0.1 --port 5003"
]

if __name__ == "__main__":
    for cmd in commands_to_execute:
        execute_command(cmd)
