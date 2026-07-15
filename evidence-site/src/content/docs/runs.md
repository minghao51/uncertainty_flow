---
title: Runs
description: Browse generated benchmark run partitions.
---

Run summaries are exported under `/evidence/runs/` as compressed JSONL partitions. The generated index is available at `/evidence/index.json`. At build time, the site validates and expands those records into static run pages at `/runs/<run-id>`.

Each record includes its run identity, status, verification flag, metrics, and artifact lineage.
