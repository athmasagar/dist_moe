#!/usr/bin/env python3
"""Regenerate numerical downloads from the article/deck's shared snapshot."""

import csv
import json
from pathlib import Path


def main() -> None:
    """Export the canonical JSON payload without executing JavaScript."""
    directory = Path(__file__).resolve().parent
    source = (directory.parent / "results-data.js").read_text()
    payload = source.split("const StandaloneData = ", 1)[1].strip().removesuffix(";")
    data = json.loads(payload)
    (directory / "standalone-results.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )
    columns = [
        "id", "mode", "ep", "category", "F", "imbalance", "precision", "pipeline",
        "forwardMS", "backwardMS", "layerMS", "usefulTFPerGPU",
        "nvlinkTXGBps", "nvlinkRXGBps", "nvlinkMaxTXGBps", "nvlinkMaxRXGBps",
        "analyticalHBMGBps", "slotBytes", "scratchBytes", "arenaBytes",
        "allocatedPeakGiB", "reservedPeakGiB", "allocationRetries", "ooms",
        "logicalForwardPayloadBytesRank0", "forwardFlopsPerLogicalPayloadByteRank0",
    ]
    with (directory / "standalone-results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["cases"])


if __name__ == "__main__":
    main()
