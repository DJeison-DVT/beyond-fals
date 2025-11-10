import re
import unicodedata
import hashlib

UNIT_KEEP = r"mg|g|kg|mcg|µg|iu|ml|l|%|bpm|mmhg|kcal|rm|vo2max|hba1c|ldl|hdl"


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", " ", s)           # collapse whitespace
    return s


def stable_id(source: str, text: str) -> str:
    h = hashlib.sha1(f"{source}::{text}".encode("utf-8")).hexdigest()[:16]
    return f"{source}-{h}"


FOOD_FITNESS_HINTS = [
    # keep short & obvious; you can expand later
    "protein", "fiber", "cholesterol", "ldl", "hdl", "hba1c", "glucose", "insulin",
    "egg", "eggs", "yogurt", "kefir", "probiotic", "creatine", "caffeine", "coffee",
    "tea", "sugar", "sweetener", "fructose", "salt", "sodium", "fat", "pufa", "omega",
    "diet", "fasting", "keto", "running", "cardio", "bench press", "1rm", "vo2max"
]


def guess_domain(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["run", "bench", "gym", "1rm", "vo2max", "athlete", "training"]):
        return "fitness"
    if any(k in t for k in FOOD_FITNESS_HINTS):
        return "food"
    return "general"
