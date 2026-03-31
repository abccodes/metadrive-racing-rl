"""Explicit train/validation/test splits for local racing maps."""

TRAIN_MAPS = [
    "circuit",
    "hairpin",
    "s_curve",
    "chicane",
    "sweeper",
]

VALIDATION_MAPS = [
    "double_hairpin",
    "technical",
]

TEST_MAPS = [
    "mixed_long",
]

SERVER_MAPS = [
    "server_map1",
    "server_map2",
    "server_map3",
    "server_map4",
]

ALL_MAPS = TRAIN_MAPS + VALIDATION_MAPS + TEST_MAPS + SERVER_MAPS

MAP_SPLITS = {
    "train": TRAIN_MAPS,
    "validation": VALIDATION_MAPS,
    "test": TEST_MAPS,
    "server": SERVER_MAPS,
    "all": ALL_MAPS,
}
