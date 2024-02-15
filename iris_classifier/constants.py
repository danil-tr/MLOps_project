import os


def get_project_path() -> str:
    current_file_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    return project_path


if __name__ == "__main__":
    pass
