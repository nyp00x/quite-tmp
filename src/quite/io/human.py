import enum


class Gender(enum.Enum):
    FEMALE = "female"
    MALE = "male"
    NEUTRAL = "neutral"


class Shot(enum.Enum):
    CLOSE_UP = "close_up"
    MEDIUM_CLOSE_UP = "medium_close_up"
    MEDIUM = "medium"
    MEDIUM_CROP = "medium_crop"
    COWBOY = "cowboy"
    COWBOY_CROP = "cowboy_crop"
    MEDIUM_WIDE = "medium_wide"
    WIDE = "wide"
    WIDE_CROP = "wide_crop"
    BOTTOM_MEDIUM = "bottom_medium"
    BOTTOM_MEDIUM_CROP = "bottom_medium_crop"
    BOTTOM_CLOSE_UP = "bottom_close_up"
