CATEGORY_LABELS = {
    "fisch": "Fisch",
    "fleisch": "Fleisch",
    "obst_gemuese": "O&G",
    "tk": "TK",
    "wurst": "Wurst",
    "mopro": "MoPro",
    "sonstiges": "Sonstiges",
}

CATEGORY_ORDER = ["fisch", "fleisch", "obst_gemuese", "tk", "wurst", "mopro", "sonstiges"]


def category_label(code: str) -> str:
    return CATEGORY_LABELS.get(code, code.title())


def category_sort_key(code: str) -> int:
    try:
        return CATEGORY_ORDER.index(code)
    except ValueError:
        return len(CATEGORY_ORDER)
