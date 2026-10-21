"""Write kneebench's bundled starter eval set: 290 questions with answers a program can check.

Run again after changing it: `uv run python scripts/make_starter_eval.py`. The output is
deterministic, so the set (and its hash) only changes when this script does.

Every answer is computed here, not typed, so the set can't hold a wrong label. It is a
smoke test for a number-format or model change, not a benchmark: use your own prompts for
the decision that matters (lesson 54 section 6).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).parents[1] / "src/kneebench/content/quality/starter.jsonl"
ONLY_NUMBER = "Reply with only the number."

CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Canada": "Ottawa",
    "Australia": "Canberra",
    "Brazil": "Brasília",
    "Egypt": "Cairo",
    "Kenya": "Nairobi",
    "India": "New Delhi",
    "Germany": "Berlin",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Mexico": "Mexico City",
    "Argentina": "Buenos Aires",
    "Norway": "Oslo",
    "Sweden": "Stockholm",
    "Poland": "Warsaw",
    "Greece": "Athens",
    "Turkey": "Ankara",
    "Thailand": "Bangkok",
    "Vietnam": "Hanoi",
    "Peru": "Lima",
    "Chile": "Santiago",
    "Portugal": "Lisbon",
    "Ireland": "Dublin",
    "Austria": "Vienna",
    "Hungary": "Budapest",
    "Finland": "Helsinki",
    "Morocco": "Rabat",
    "Nigeria": "Abuja",
    "South Korea": "Seoul",
}
SYMBOLS = {
    "hydrogen": "H",
    "helium": "He",
    "carbon": "C",
    "nitrogen": "N",
    "oxygen": "O",
    "sodium": "Na",
    "magnesium": "Mg",
    "aluminium": "Al",
    "silicon": "Si",
    "sulfur": "S",
    "chlorine": "Cl",
    "potassium": "K",
    "calcium": "Ca",
    "iron": "Fe",
    "copper": "Cu",
    "zinc": "Zn",
    "silver": "Ag",
    "gold": "Au",
    "lead": "Pb",
    "mercury": "Hg",
}
NAMES = ["Asha", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goran", "Hana", "Ivan", "Jun"]
CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Hamburg", "Pune", "Quito", "Perth"]
WORDS = [
    "garden",
    "planet",
    "silver",
    "rocket",
    "window",
    "pencil",
    "candle",
    "bridge",
    "forest",
    "button",
    "marble",
    "ladder",
    "basket",
    "violin",
    "anchor",
    "copper",
]


def arithmetic(rng: random.Random) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for _i in range(40):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        items.append(_q(f"What is {a} × {b}? {ONLY_NUMBER}", a * b, "number", "arithmetic"))
    for _i in range(30):
        a, b, c = rng.randint(100, 999), rng.randint(100, 999), rng.randint(10, 99)
        items.append(
            _q(f"What is {a} + {b} - {c}? {ONLY_NUMBER}", a + b - c, "number", "arithmetic")
        )
    for _i in range(30):
        boxes, per, extra = rng.randint(3, 25), rng.randint(6, 48), rng.randint(5, 90)
        name = rng.choice(NAMES)
        prompt = (
            f"{name} packs {boxes} boxes with {per} pens each, then adds {extra} loose pens. "
            f"How many pens are there in total? {ONLY_NUMBER}"
        )
        items.append(_q(prompt, boxes * per + extra, "number", "arithmetic"))
    return items


def extraction(rng: random.Random) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for i in range(80):
        name = rng.choice(NAMES)
        city = rng.choice(CITIES)
        qty, price = rng.randint(2, 40), rng.randint(3, 250)
        number = rng.randint(10_000, 99_999)
        day, month = rng.randint(1, 28), rng.choice(["March", "June", "October", "December"])
        email = f"{name.lower()}.{rng.choice(WORDS)}@example.com"
        text = (
            f"Invoice {number}, issued {day} {month} 2024 in {city}. Customer: {name}, "
            f"contact {email}. Items: {qty} units at {price} dollars each."
        )
        ask = i % 4
        if ask == 0:
            items.append(
                _q(
                    f"{text}\n\nWhat is the invoice total in dollars? {ONLY_NUMBER}",
                    qty * price,
                    "number",
                    "extraction",
                )
            )
        elif ask == 1:
            items.append(
                _q(
                    f"{text}\n\nWhat is the customer's email address? Reply with only the address.",
                    email,
                    "contains",
                    "extraction",
                )
            )
        elif ask == 2:
            items.append(
                _q(
                    f"{text}\n\nWhat is the invoice number? {ONLY_NUMBER}",
                    number,
                    "number",
                    "extraction",
                )
            )
        else:
            items.append(
                _q(
                    f"{text}\n\nIn which city was the invoice issued? Reply with only "
                    "the city name.",
                    city,
                    "exact",
                    "extraction",
                )
            )
    return items


def instructions(rng: random.Random) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for _i in range(20):
        word = rng.choice(WORDS)
        items.append(
            _q(
                f"Write the word '{word}' backwards. Reply with only the result.",
                word[::-1],
                "exact",
                "instructions",
            )
        )
    for _i in range(20):
        nums = rng.sample(range(1, 100), 5)
        items.append(
            _q(
                f"Sort these numbers from smallest to largest: {', '.join(map(str, nums))}. Reply "
                "with only the sorted numbers, separated by commas.",
                ", ".join(map(str, sorted(nums))),
                "exact",
                "instructions",
            )
        )
    for _i in range(20):
        word = rng.choice(WORDS)
        items.append(
            _q(
                f"Reply with the word '{word}' in capital letters and nothing else.",
                word.upper(),
                "exact_case",
                "instructions",
            )
        )
    return items


def knowledge(rng: random.Random) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for table, template in (
        (CAPITALS, "What is the capital of {}?"),
        (SYMBOLS, "What is the chemical symbol for {}?"),
    ):
        keys = list(table)
        for key in keys:
            wrong = rng.sample([table[k] for k in keys if k != key], 3)
            options = [table[key], *wrong]
            rng.shuffle(options)
            letter = "ABCD"[options.index(table[key])]
            lines = "\n".join(f"{'ABCD'[j]}. {o}" for j, o in enumerate(options))
            prompt = f"{template.format(key)}\n{lines}\n\nReply with only the letter."
            items.append(_q(prompt, letter, "choice", "knowledge"))
    return items


def _q(prompt: str, answer: object, match: str, category: str) -> dict[str, object]:
    return {"prompt": prompt, "answer": str(answer), "match": match, "category": category}


def main() -> None:
    rng = random.Random(54)
    items = arithmetic(rng) + extraction(rng) + instructions(rng) + knowledge(rng)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for i, item in enumerate(items):
            fh.write(json.dumps({"id": f"s{i:03d}", **item}, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} questions to {OUT}")


if __name__ == "__main__":
    main()
