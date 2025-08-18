from enum import IntEnum, Enum


class Drafted(IntEnum):
    NONE = 0
    OTHER = 1
    MINE = 2


class Sport(str, Enum):
    NFL = "nfl"
    NBA = "nba"

