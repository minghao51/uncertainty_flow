# Uncertainty Flow Evidence Site

This is a separate Astro/Starlight site for generated Platinum evidence. It is intentionally isolated from the repository's existing MkDocs `site/` output.

Before building, export evidence into this project's public directory:

```bash
uv run uncertainty-flow pipeline export-site \
  --root ../data \
  --output public/evidence
npm install
npm run build
```

The build consumes the small `public/evidence/index.json` catalog and links to compressed run partitions. Prediction-level Parquet artifacts remain outside the static site.
