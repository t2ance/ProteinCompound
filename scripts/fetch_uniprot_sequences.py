#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def _default_paths(input_path: str) -> Tuple[str, str, str, str]:
    base, _ext = os.path.splitext(input_path)
    out_csv = f"{base}_with_sequences.csv"
    mapping_tsv = f"{base}_uniprot_mapping.tsv"
    cache_json = f"{base}_uniprot_cache.json"
    unmatched_txt = f"{base}_unmatched.txt"
    return out_csv, mapping_tsv, cache_json, unmatched_txt


def _escape_gene(gene: str) -> str:
    return gene.replace("\\", "\\\\").replace('"', '\\"')


def _build_query(genes: Iterable[str], taxon: str) -> str:
    parts = [f'gene_exact:"{_escape_gene(g)}"' for g in genes]
    return f"({' OR '.join(parts)}) AND organism_id:{taxon}"


def _request_with_backoff(
    url: str,
    params: Optional[Dict[str, str]],
    max_retries: int,
    sleep_seconds: float,
    timeout: int,
) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    time.sleep(float(retry_after))
                else:
                    time.sleep(sleep_seconds * (2 ** attempt))
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(sleep_seconds * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to contact UniProt.")


def _extract_next_link(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    parts = link_header.split(",")
    for part in parts:
        section = part.strip().split(";")
        if len(section) < 2:
            continue
        url_part = section[0].strip()
        rel_part = section[1].strip()
        if rel_part == 'rel="next"' and url_part.startswith("<") and url_part.endswith(">"):
            return url_part[1:-1]
    return None


def _parse_tsv(text: str) -> List[Dict[str, str]]:
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    return list(reader)


def _pick_better(current: Optional[Dict[str, str]], candidate: Dict[str, str]) -> Dict[str, str]:
    if current is None:
        return candidate
    cur_reviewed = current.get("reviewed") == "reviewed"
    cand_reviewed = candidate.get("reviewed") == "reviewed"
    if cand_reviewed and not cur_reviewed:
        return candidate
    if cand_reviewed == cur_reviewed:
        if len(candidate.get("sequence", "")) > len(current.get("sequence", "")):
            return candidate
    return current


def _load_cache(cache_path: str, taxon: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    if cache.get("taxon") != str(taxon):
        return {}
    return cache.get("entries", {})


def _save_cache(cache_path: str, taxon: str, entries: Dict[str, Dict[str, str]]) -> None:
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"taxon": str(taxon), "entries": entries}, f)
    os.replace(tmp_path, cache_path)


def _fetch_batch(
    genes: List[str],
    taxon: str,
    max_retries: int,
    sleep_seconds: float,
    timeout: int,
) -> Dict[str, Dict[str, str]]:
    fields = "accession,sequence,protein_name,reviewed,gene_primary,gene_names"
    query = _build_query(genes, taxon)
    params = {"query": query, "format": "tsv", "fields": fields, "size": "500"}
    rows: List[Dict[str, str]] = []
    next_url: Optional[str] = UNIPROT_SEARCH_URL
    next_params: Optional[Dict[str, str]] = params
    while next_url:
        resp = _request_with_backoff(next_url, next_params, max_retries, sleep_seconds, timeout)
        if not resp.ok:
            raise RuntimeError(f"UniProt request failed: {resp.status_code} {resp.text[:200]}")
        rows.extend(_parse_tsv(resp.text))
        next_url = _extract_next_link(resp.headers.get("Link"))
        next_params = None
    batch_set = set(genes)
    batch_entries: Dict[str, Dict[str, str]] = {}
    for row in rows:
        gene_names = row.get("Gene Names", "") or ""
        names = [n for n in gene_names.split() if n]
        matched = [g for g in names if g in batch_set]
        if not matched:
            continue
        candidate = {
            "accession": row.get("Entry", ""),
            "sequence": row.get("Sequence", ""),
            "reviewed": row.get("Reviewed", ""),
            "protein_name": row.get("Protein names", ""),
        }
        for gene in matched:
            batch_entries[gene] = _pick_better(batch_entries.get(gene), candidate)
    return batch_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch UniProt sequences for gene symbols.")
    parser.add_argument(
        "--input",
        default="datasets/SMOPINs_openDel_3_novaSeq_SMILES.csv",
        help="Input CSV with a Protein column.",
    )
    parser.add_argument("--protein-col", default="Protein", help="Column with gene symbols.")
    parser.add_argument("--taxon", default="9606", help="NCBI taxonomy ID (default: human 9606).")
    parser.add_argument("--batch-size", type=int, default=200, help="Genes per UniProt query.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Base sleep for retries.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per batch.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds.")
    parser.add_argument("--output", default=None, help="Output CSV with sequences.")
    parser.add_argument("--mapping-out", default=None, help="TSV mapping of gene to UniProt.")
    parser.add_argument("--cache", default=None, help="Cache JSON to resume.")
    parser.add_argument("--unmatched", default=None, help="Unmatched genes list.")
    args = parser.parse_args()

    out_csv, mapping_tsv, cache_json, unmatched_txt = _default_paths(args.input)
    if args.output is None:
        args.output = out_csv
    if args.mapping_out is None:
        args.mapping_out = mapping_tsv
    if args.cache is None:
        args.cache = cache_json
    if args.unmatched is None:
        args.unmatched = unmatched_txt

    df = pd.read_csv(args.input)
    if args.protein_col not in df.columns:
        raise ValueError(f"Missing column: {args.protein_col}")
    proteins = df[args.protein_col].astype(str).tolist()
    unique_genes = sorted(set(proteins))

    entries = _load_cache(args.cache, args.taxon)
    missing = [g for g in unique_genes if g not in entries]

    for i in range(0, len(missing), args.batch_size):
        batch = missing[i : i + args.batch_size]
        batch_entries = _fetch_batch(
            batch,
            args.taxon,
            args.max_retries,
            args.sleep,
            args.timeout,
        )
        entries.update(batch_entries)
        _save_cache(args.cache, args.taxon, entries)
        time.sleep(args.sleep)

    unmatched = [g for g in unique_genes if g not in entries]
    if unmatched:
        with open(args.unmatched, "w", encoding="utf-8") as f:
            for g in unmatched:
                f.write(f"{g}\n")

    df["uniprot_accession"] = df[args.protein_col].map(
        lambda g: entries.get(str(g), {}).get("accession")
    )
    df["protein_sequence"] = df[args.protein_col].map(
        lambda g: entries.get(str(g), {}).get("sequence")
    )
    df.to_csv(args.output, index=False)

    with open(args.mapping_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Protein", "UniProtAccession", "Reviewed", "ProteinName", "Sequence"])
        for gene in sorted(entries.keys()):
            entry = entries[gene]
            writer.writerow(
                [
                    gene,
                    entry.get("accession", ""),
                    entry.get("reviewed", ""),
                    entry.get("protein_name", ""),
                    entry.get("sequence", ""),
                ]
            )


if __name__ == "__main__":
    main()
