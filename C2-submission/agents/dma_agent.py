#!/usr/bin/env python3

import json
import sys


MAX_CHUNK_BYTES = 1024 * 1024


def choose_queue_depth(concurrency: int) -> int:
    # parallelism=min(queue_depth, concurrency, 2)
    return 2 if concurrency >= 2 else 1


def main() -> None:
    request = json.load(sys.stdin)

    concurrency = max(
        1,
        int(request["concurrency"]),
    )

    direction = str(
        request["direction"]
    ).lower()

    registered = bool(
        request["registered"]
    )

    action = {
        "channel": (
            0 if direction == "h2d" else 1
        ),
        "chunk_bytes": MAX_CHUNK_BYTES,
        "queue_depth": choose_queue_depth(
            concurrency
        ),
        "use_zero_copy": registered,
    }

    json.dump(
        action,
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
    )

    sys.stdout.write("\n")


if __name__ == "__main__":
    main()