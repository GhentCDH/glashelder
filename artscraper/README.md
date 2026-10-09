# Artscraper

Async scraper that builds an index of externally described artworks to match glass
plate photos against.

## Commands

```bash
# scrape the configured backends into a JSON list of artworks
uv run glashelder artscraper scrape

# combine one or more jsonl or json files of artworks into one
uv run glashelder artscraper combine_jsons
```
