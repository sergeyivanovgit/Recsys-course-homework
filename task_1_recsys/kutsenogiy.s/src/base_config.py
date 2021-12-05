from dataclasses import dataclass
import typing as tp


@dataclass
class Config:
    data_path: str
    size: int
    train_start_date: str
    train_end_date: str
    test_date: str
    features_to_generate: tp.List[tp.Dict]
    features_to_train: tp.List[str]
    no_matter_features: tp.Set
    random_state: int
