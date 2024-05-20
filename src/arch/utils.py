import yaml
from src.utils.constants import SETUP_FILE

def read_setup(arch_setup, bases=None):
    if bases is None:
        bases = set()

    with open(SETUP_FILE, "r") as yaml_file:
        arch_setup_data = yaml.safe_load(yaml_file)[arch_setup]
        bases.add(arch_setup)

    if "base" not in arch_setup_data.keys():
        return arch_setup_data
    if arch_setup_data["base"] in bases:
        return arch_setup_data
    else:
        arch_base_data = read_setup(arch_setup_data["base"], bases)

        arch_base_data.update(arch_setup_data)

        return arch_base_data


    
    