import os


def get_env_var(var: str) -> str:
    value = os.environ.get(var)
    if value is None:
        print(f"Environment variable {var} is not set.")
    return os.environ.get(var)
