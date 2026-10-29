# LeCaRDv2 local data (optional)

LexCaR can run with `--demo` or `--query` without any local dataset.

To use `--qid` (sample facts from LeCaRDv2), place files here:

```
data/lecardv2/
  query.json          # one JSON object per line (id + fact/query)
  candidates.jsonl    # optional; only needed for local candidate loading helpers
```

## Where to get the data

Official release: [https://github.com/THUIR/LeCaRDv2](https://github.com/THUIR/LeCaRDv2)

You may also load subsets via Hugging Face (`mteb/LeCaRDv2`) through `data/lecardv2.py` when networking is available.

## License

LeCaRDv2 is **not** redistributed in this repository. Respect the upstream dataset license and terms of use.
