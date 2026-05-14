from datetime import date, timedelta


def current_week() -> tuple[int, int]:
    today = date.today()
    iso = today.isocalendar()
    return iso.week, iso.year


def next_week() -> tuple[int, int]:
    next_monday = date.today() + timedelta(days=7 - date.today().weekday())
    iso = next_monday.isocalendar()
    return iso.week, iso.year


def week_date_range(week: int, year: int) -> tuple[date, date]:
    monday = date.fromisocalendar(year, week, 1)
    sunday = monday + timedelta(days=6)
    return monday, sunday


def format_week(week: int, year: int) -> str:
    return f"KW{week:02d}_{year}"


def week_dir(supplier: str, week: int, year: int) -> str:
    return f"{supplier}/{year}/{week:02d}"
