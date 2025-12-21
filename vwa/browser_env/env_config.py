# NOTE[mandrade]: substantially refactored to support dynamic URLs, website display names, env parallelization

from enum import StrEnum

from benchmark_config import URLS


class Website(StrEnum):
    HOMEPAGE = "homepage"
    SHOPPING = "shopping"
    CLASSIFIEDS = "classifieds"
    REDDIT = "reddit"
    WIKIPEDIA = "wikipedia"
    SHOPPING_ADMIN = "shopping_admin"
    GITLAB = "gitlab"
    MAP = "map"


URL_MAPPINGS = {
    Website.HOMEPAGE: URLS["homepage"],
    Website.SHOPPING: URLS["shopping"],
    Website.REDDIT: URLS["reddit"],
    Website.WIKIPEDIA: URLS["wikipedia"],
    Website.CLASSIFIEDS: URLS["classifieds"],
    Website.SHOPPING_ADMIN: URLS["shopping_admin"],
    Website.GITLAB: URLS["gitlab"],
    Website.MAP: URLS["map"],
}


ACCOUNTS = {
    Website.REDDIT: {"username": "MarvelsGrantMan136", "password": "test1234"},
    Website.SHOPPING: {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    Website.CLASSIFIEDS: {
        "username": "blake.sullivan@gmail.com",
        "password": "Password.123",
    },
    Website.SHOPPING_ADMIN: {"username": "admin", "password": "admin1234"},
    Website.GITLAB: {"username": "byteblaze", "password": "hello1234"},
}

SITE_DISPLAY_NAMES = {
    # e.g.: http://forum -> FORUM; http://reddit -> REDDIT
    Website.REDDIT: URL_MAPPINGS[Website.REDDIT][0].split("//")[1].split(".")[0].capitalize(),
    Website.SHOPPING: "One Stop Market",
    Website.HOMEPAGE: "Homepage",
    Website.CLASSIFIEDS: URL_MAPPINGS[Website.CLASSIFIEDS][0].split("//")[1].split(".")[0].capitalize(),
    Website.GITLAB: "GitLab",
    Website.MAP: "OpenStreetMap",
    Website.WIKIPEDIA: "Wikipedia",
    Website.SHOPPING_ADMIN: "Luma Admin",
}


PAGE_NAME_NORMALIZATIONS = {
    Website.WIKIPEDIA: ("User:The other Kiwix guy/Landing", "Welcome to Wikipedia"),
}

VWA_DOMAINS = [
    Website.HOMEPAGE,
    Website.SHOPPING,
    Website.REDDIT,
    Website.WIKIPEDIA,
    Website.CLASSIFIEDS,
]

WA_DOMAINS = [
    Website.HOMEPAGE,
    Website.SHOPPING,
    Website.REDDIT,
    Website.SHOPPING_ADMIN,
    Website.GITLAB,
    Website.MAP,
    Website.WIKIPEDIA,
]

NO_COOKIE_RESET_SITES = [Website.WIKIPEDIA, Website.HOMEPAGE, Website.MAP]


def get_username(website: Website) -> str:
    """Get the username for a given website."""
    username = ACCOUNTS[website]["username"]
    return username


def get_username_and_password(website: Website) -> tuple[str, str]:
    """Get the username and password for a given website."""
    username = ACCOUNTS[website]["username"]
    password = ACCOUNTS[website]["password"]
    return username, password
