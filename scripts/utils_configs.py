from cfnet.utils import load_json
from typing import List, Dict, Any


configs = {
    "adult": "assets/configs/data_configs/adult.json",
    "home": "assets/configs/data_configs/home.json",
    "student": "assets/configs/data_configs/student.json",
    "breast_cancer": "assets/configs/data_configs/breast_cancer.json",
    "student performance": "assets/configs/data_configs/student_performance.json",
    "titanic": "assets/configs/data_configs/titanic.json",
    "credit": "assets/configs/data_configs/credit_card.json",
    "german": "assets/configs/data_configs/german_credit.json"
}


def get_configs(data_name: str) -> List[Dict[str, Any]]:
    if data_name == "all":
        return [
            load_json(f_name) for _, f_name in configs.items()
        ]
    
    if data_name not in configs.keys():
        raise ValueError(f"`data_name` should be one of `{configs.keys()}`, but got `{data_name}`")
    
    return [ load_json(configs[data_name]) ]
