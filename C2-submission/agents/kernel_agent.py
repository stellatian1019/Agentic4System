#!/usr/bin/env python3

import json
import sys


DIVISIBILITY = {1: 1, 2: 4, 3: 8}
ALIGNMENT = {1: 1, 2: 1, 3: 16}


def is_legal(candidate: dict, request: dict) -> bool:
    variant = candidate["variant"]

    divisibility = max(
        candidate["divisibility"],
        DIVISIBILITY[variant],
    )

    alignment = max(
        candidate["alignment"],
        ALIGNMENT[variant],
    )

    return (
        request["alignment"] >= alignment
        and request["workspace"] >= candidate["workspace"]
        and request["m"] % divisibility == 0
        and request["n"] % divisibility == 0
        and request["k"] % divisibility == 0
    )


def main() -> None:
    request = json.load(sys.stdin)

    legal = [
        candidate
        for candidate in request["candidates"]
        if is_legal(candidate, request)
    ]

    if not legal:
        raise SystemExit(1)

    selected = max(
        legal,
        key=lambda candidate: candidate["variant"],
    )

    json.dump(
        {"kernel_id": selected["id"]},
        sys.stdout,
        separators=(",", ":"),
    )

    sys.stdout.write("\n")


if __name__ == "__main__":
    main()