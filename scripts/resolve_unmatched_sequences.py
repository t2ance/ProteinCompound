#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


ENSEMBL_LOOKUP_URL = "https://rest.ensembl.org/lookup/symbol"
ENSEMBL_SEQUENCE_URL = "https://rest.ensembl.org/sequence/id"
NCBI_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _default_paths(input_path: str) -> Tuple[str, str, str, str]:
    base, _ext = os.path.splitext(input_path)
    out_csv = f"{base}_resolved.csv"
    mapping_tsv = f"{base}_fallback_mapping.tsv"
    cache_json = f"{base}_fallback_cache.json"
    unmatched_txt = f"{base}_unmatched_after_fallback.txt"
    return out_csv, mapping_tsv, cache_json, unmatched_txt


def _request_with_backoff(
    url: str,
    params: Optional[Dict[str, str]],
    headers: Optional[Dict[str, str]],
    max_retries: int,
    sleep_seconds: float,
    timeout: int,
) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
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
    raise RuntimeError("Failed to contact external service.")


def _load_cache(cache_path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(cache_path: str, entries: Dict[str, Dict[str, str]]) -> None:
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    os.replace(tmp_path, cache_path)


def _ensembl_lookup(
    symbol: str,
    species: str,
    max_retries: int,
    sleep_seconds: float,
    timeout: int,
) -> Optional[Dict[str, str]]:
    url = f"{ENSEMBL_LOOKUP_URL}/{species}/{symbol}"
    params = {"expand": "1"}
    headers = {"Content-Type": "application/json"}
    resp = _request_with_backoff(url, params, headers, max_retries, sleep_seconds, timeout)
    if resp.status_code == 404:
        return None
    if not resp.ok:
        return None
    data = resp.json()
    transcripts = data.get("Transcript", []) or []
    canonical_id = data.get("canonical_transcript")
    translation_id = None
    if canonical_id:
        for transcript in transcripts:
            if transcript.get("id") == canonical_id:
                translation_id = (transcript.get("Translation") or {}).get("id")
                break
    if translation_id is None:
        for transcript in transcripts:
            translation_id = (transcript.get("Translation") or {}).get("id")
            if translation_id:
                break
    if not translation_id:
        return None
    seq = _ensembl_sequence(translation_id, max_retries, sleep_seconds, timeout)
    if not seq:
        return None
    return {
        "source": "ensembl",
        "accession": translation_id,
        "sequence": seq,
        "gene_id": data.get("id", ""),
    }


def _ensembl_sequence(
    translation_id: str,
    max_retries: int,
    sleep_seconds: float,
    timeout: int,
) -> Optional[str]:
    url = f"{ENSEMBL_SEQUENCE_URL}/{translation_id}"
    params = {"type": "protein"}
    headers = {"Content-Type": "application/json"}
    resp = _request_with_backoff(url, params, headers, max_retries, sleep_seconds, timeout)
    if not resp.ok:
        return None
    data = resp.json()
    return data.get("seq")


def _ncbi_esearch_gene(
    symbol: str,
    email: Optional[str],
    max_retries: int,
    sleep_seconds: float,
    timeout: int,
) -> Optional[str]:
    term = f'{symbol}[sym] AND "Homo sapiens"[orgn]'
    params = {"db": "gene", "term": term, "retmode": "json"}
    if email:
        params["email"] = email
    url = f"{NCBI_EUTILS_URL}/esearch.fcgi"
    resp = _request_with_backoff(url, params, None, max_retries, sleep_seconds, timeout)
    if not resp.ok:
        return None
    data = resp.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


def _ncbi_elink_protein(
    gene_id: str,
    email: Optional[str],
    max_retries: int,
    sleep_seconds: float,
    timeout: int,
) -> List[str]:
    params = {"dbfrom": "gene", "db": "protein", "id": gene_id, "retmode": "json"}
    if email:
        params["email"] = email
    url = f"{NCBI_EUTILS_URL}/elink.fcgi"
    resp = _request_with_backoff(url, params, None, max_retries, sleep_seconds, timeout)
    if not resp.ok:
        return []
    data = resp.json()
    links = data.get("linksets", [])
    for linkset in links:
        for db in linkset.get("linksetdbs", []):
            ids = db.get("links", [])
            if ids:
                return [str(i) for i in ids]
    return []


def _parse_fasta(text: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_parts: List[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header:
                records.append((_extract_accession(header), "".join(seq_parts)))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    if header:
        records.append((_extract_accession(header), "".join(seq_parts)))
    return records


def _extract_accession(header: str) -> str:
    token = header.split()[0]
    if "|" in token:
        parts = token.split("|")
        for i, part in enumerate(parts):
            if part in ("ref", "sp", "gb", "emb", "dbj"):
                if i + 1 < len(parts):
                    return parts[i + 1]
        return parts[-1]
    return token


def _pick_best_ncbi(records: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    if not records:
        return None
    def rank(acc: str) -> int:
        if acc.startswith("NP_"):
            return 0
        if acc.startswith("XP_"):
            return 1
        return 2
    best = None
    for acc, seq in records:
        if not seq:
            continue
        if best is None:
            best = (acc, seq)
            continue
        if rank(acc) < rank(best[0]):
            best = (acc, seq)
        elif rank(acc) == rank(best[0]) and len(seq) > len(best[1]):
            best = (acc, seq)
    return best


def _ncbi_fetch_protein(
    protein_ids: List[str],
    email: Optional[str],
    max_retries: int,
    sleep_seconds: float,
    timeout: int,
) -> Optional[Tuple[str, str]]:
    if not protein_ids:
        return None
    params = {"db": "protein", "id": ",".join(protein_ids), "rettype": "fasta", "retmode": "text"}
    if email:
        params["email"] = email
    url = f"{NCBI_EUTILS_URL}/efetch.fcgi"
    resp = _request_with_backoff(url, params, None, max_retries, sleep_seconds, timeout)
    if not resp.ok:
        return None
    records = _parse_fasta(resp.text)
    return _pick_best_ncbi(records)


def _ncbi_lookup(
    symbol: str,
    email: Optional[str],
    max_retries: int,
    sleep_seconds: float,
    timeout: int,
) -> Optional[Dict[str, str]]:
    gene_id = _ncbi_esearch_gene(symbol, email, max_retries, sleep_seconds, timeout)
    if not gene_id:
        return None
    protein_ids = _ncbi_elink_protein(gene_id, email, max_retries, sleep_seconds, timeout)
    best = _ncbi_fetch_protein(protein_ids, email, max_retries, sleep_seconds, timeout)
    if not best:
        return None
    acc, seq = best
    return {"source": "ncbi", "accession": acc, "sequence": seq, "gene_id": gene_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve missing protein sequences via Ensembl/NCBI.")
    parser.add_argument(
        "--input",
        default="datasets/SMOPINs_openDel_3_novaSeq_SMILES_with_sequences.csv",
        help="Input CSV with protein_sequence column.",
    )
    parser.add_argument(
        "--unmatched",
        default="datasets/SMOPINs_openDel_3_novaSeq_SMILES_unmatched.txt",
        help="Unmatched gene symbols list.",
    )
    parser.add_argument("--output", default=None, help="Output CSV with resolved sequences.")
    parser.add_argument("--mapping-out", default=None, help="Fallback mapping TSV.")
    parser.add_argument("--cache", default=None, help="Cache JSON for fallbacks.")
    parser.add_argument("--unmatched-out", default=None, help="Remaining unmatched list.")
    parser.add_argument("--ensembl-species", default="homo_sapiens", help="Ensembl species name.")
    parser.add_argument("--sleep", type=float, default=0.34, help="Sleep between requests.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")
    parser.add_argument("--ncbi-email", default=None, help="Email for NCBI requests.")
    parser.add_argument(
        "--label-output",
        default=None,
        help="Output CSV with rna_sequence, smiles_sequence, label columns.",
    )
    args = parser.parse_args()

    out_csv, mapping_tsv, cache_json, unmatched_txt = _default_paths(args.input)
    if args.output is None:
        args.output = out_csv
    if args.mapping_out is None:
        args.mapping_out = mapping_tsv
    if args.cache is None:
        args.cache = cache_json
    if args.unmatched_out is None:
        args.unmatched_out = unmatched_txt
    if args.label_output is None:
        base, _ext = os.path.splitext(args.output)
        args.label_output = f"{base}_rna_smiles_labels.csv"

    with open(args.unmatched, "r", encoding="utf-8") as f:
        symbols = [line.strip() for line in f if line.strip()]

    cache = _load_cache(args.cache)
    resolved: Dict[str, Dict[str, str]] = cache.get("entries", {}) if cache else {}
    unresolved = []

    for symbol in symbols:
        if symbol in resolved:
            continue
        entry = _ensembl_lookup(
            symbol,
            args.ensembl_species,
            args.max_retries,
            args.sleep,
            args.timeout,
        )
        time.sleep(args.sleep)
        if entry is None:
            entry = _ncbi_lookup(
                symbol,
                args.ncbi_email,
                args.max_retries,
                args.sleep,
                args.timeout,
            )
            time.sleep(args.sleep)
        if entry is None:
            unresolved.append(symbol)
            continue
        resolved[symbol] = entry
        cache = {"entries": resolved}
        _save_cache(args.cache, cache)

    df = pd.read_csv(args.input)
    if "sequence_source" not in df.columns:
        df["sequence_source"] = None
    if "sequence_accession" not in df.columns:
        df["sequence_accession"] = None

    def fill_sequence(gene: str, current_seq: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        gene = str(gene)
        if pd.notna(current_seq):
            return current_seq, "uniprot", None
        entry = resolved.get(gene)
        if not entry:
            return current_seq, None, None
        return entry.get("sequence"), entry.get("source"), entry.get("accession")

    seqs = []
    sources = []
    accessions = []
    for gene, current_seq in zip(df["Protein"], df["protein_sequence"]):
        seq, source, accession = fill_sequence(gene, current_seq)
        seqs.append(seq)
        sources.append(source)
        accessions.append(accession)
    df["protein_sequence"] = seqs
    df["sequence_source"] = sources
    df["sequence_accession"] = accessions
    df.to_csv(args.output, index=False)

    label_df = pd.DataFrame(
        {
            "rna_sequence": df["protein_sequence"].fillna(""),
            "smiles_sequence": df["smilesStructure"].fillna(""),
            "label": 1,
        }
    )
    label_df.to_csv(args.label_output, index=False)

    with open(args.mapping_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Protein", "Source", "Accession", "GeneId", "Sequence"])
        for gene in sorted(resolved.keys()):
            entry = resolved[gene]
            writer.writerow(
                [
                    gene,
                    entry.get("source", ""),
                    entry.get("accession", ""),
                    entry.get("gene_id", ""),
                    entry.get("sequence", ""),
                ]
            )

    remaining = [s for s in symbols if s not in resolved]
    if remaining:
        with open(args.unmatched_out, "w", encoding="utf-8") as f:
            for gene in remaining:
                f.write(f"{gene}\n")


if __name__ == "__main__":
    main()
