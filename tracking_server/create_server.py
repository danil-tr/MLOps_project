import subprocess


def execute_command(command):
    subprocess.run(command, shell=True, check=True)


commands_to_execute = ["sudo docker compose up -d --build"]


if __name__ == "__main__":
    for cmd in commands_to_execute:
        execute_command(cmd)
