from src.research.dedup import dedupe_batch, find_duplicate, identity_keys, merge_metadata
from src.research.models import NormalizedPaper
from src.research.normalize import (
    canonical_url, clean_arxiv_id, clean_doi, normalize_searxng_result, normalize_title,
)


# ── identifiers ──

def test_clean_doi_from_url_and_prefix():
    assert clean_doi("https://doi.org/10.1145/3597503.3639180") == "10.1145/3597503.3639180"
    assert clean_doi("doi:10.1000/XYZ.123.") == "10.1000/xyz.123"
    assert clean_doi("https://dl.acm.org/doi/pdf/10.1145/123.456") == "10.1145/123.456"
    assert clean_doi("no doi here") == ""


def test_clean_arxiv_id_strips_version_and_handles_pdf_urls():
    assert clean_arxiv_id("https://arxiv.org/abs/2305.04207v3") == "2305.04207"
    assert clean_arxiv_id("https://arxiv.org/pdf/2305.04207v1.pdf") == "2305.04207"
    assert clean_arxiv_id("arXiv:2101.00001") == "2101.00001"
    assert clean_arxiv_id("https://arxiv.org/abs/cs/0112017") == "cs/0112017"
    assert clean_arxiv_id("https://example.org/paper") == ""


def test_canonical_url_drops_tracking_and_www():
    assert canonical_url("http://www.Example.org/paper/?utm_source=x&id=3#top") == "https://example.org/paper?id=3"
    assert canonical_url("ftp://example.org/x") == ""


def test_normalize_title():
    assert normalize_title("  <b>Déjà Vu</b>: LLMs & Tests! ") == "deja vu llms tests"


# ── SearXNG normalisation ──

def test_normalize_paper_template_result():
    raw = {
        "title": "An <span>Empirical</span> Study", "url": "https://arxiv.org/abs/2305.00001v2",
        "content": "We study things.", "engine": "arxiv", "engines": ["arxiv", "semantic scholar"],
        "publishedDate": "2023-05-01T00:00:00", "authors": ["A. Smith", "B. Lee"], "journal": "ICSE",
        "template": "paper.html",
    }
    p = normalize_searxng_result(raw, "q")
    assert p.title == "An Empirical Study"
    assert p.arxiv_id == "2305.00001"
    assert p.pdf_url == "https://arxiv.org/pdf/2305.00001"
    assert p.year == 2023 and p.publication_date == "2023-05-01"
    assert p.authors == ["A. Smith", "B. Lee"]
    assert p.venue == "ICSE"
    assert p.engines == ["arxiv", "semantic scholar"]
    assert p.academic_signal and p.found_by_query == "q"


def test_normalize_handles_missing_metadata_and_general_results():
    p = normalize_searxng_result({"title": "Some blog post", "url": "https://blog.example.com/post"})
    assert p.abstract == "" and p.doi == "" and p.year is None and p.authors == []
    assert not p.academic_signal
    assert normalize_searxng_result({"title": "", "url": "https://x.org"}) is None
    assert normalize_searxng_result({"title": "No url"}) is None


def test_normalize_extracts_doi_from_url_and_string_authors():
    p = normalize_searxng_result({"title": "T", "url": "https://doi.org/10.1145/1.2", "authors": "Ann Lee; Bo Wu",
                                  "citations": "12"})
    assert p.doi == "10.1145/1.2"
    assert p.authors == ["Ann Lee", "Bo Wu"]
    assert p.citation_count == 12


# ── deduplication ──

def _paper(**kw):
    base = {"title": "Large Language Models for Unit Test Generation", "authors": [], "url": "", "doi": "",
            "arxiv_id": "", "pdf_url": ""}
    return {**base, **kw}


def test_duplicate_by_doi_across_sites():
    existing = [_paper(id="1", url="https://dl.acm.org/doi/10.1145/1.2", doi="10.1145/1.2", title="Title A")]
    cand = _paper(url="https://doi.org/10.1145/1.2", title="Completely different title text")
    assert find_duplicate(cand, existing)["id"] == "1"


def test_duplicate_by_arxiv_id_different_versions():
    existing = [_paper(id="1", url="https://arxiv.org/abs/2305.00001v1", title="X" * 30)]
    cand = _paper(url="https://arxiv.org/pdf/2305.00001v3.pdf", title="Y" * 30)
    assert find_duplicate(cand, existing)["id"] == "1"


def test_duplicate_by_fuzzy_title_and_author():
    existing = [_paper(id="1", title="Large Language Models for Unit Test Generation: An Empirical Study",
                       authors=["Alice Smith"])]
    cand = _paper(title="Large language models for unit-test generation - an empirical study",
                  authors=["A. Smith"])
    assert find_duplicate(cand, existing)["id"] == "1"


def test_not_duplicate_when_dois_conflict():
    existing = [_paper(id="1", doi="10.1000/aaa")]
    cand = _paper(doi="10.1000/bbb")  # same title, different DOI → different papers (e.g. journal vs erratum)
    assert find_duplicate(cand, existing) is None


def test_short_titles_are_not_matched_on_title_alone():
    existing = [_paper(id="1", title="Introduction")]
    assert find_duplicate(_paper(title="Introduction"), existing) is None
    assert not any(k.startswith("title:") for k in identity_keys(_paper(title="Intro")))


def test_dedupe_batch_merges_metadata():
    a = NormalizedPaper(title="Large Language Models for Unit Test Generation", url="https://arxiv.org/abs/2305.00001")
    b = NormalizedPaper(title="Large Language Models for Unit Test Generation",
                        url="https://www.semanticscholar.org/paper/x", doi="10.1000/abc", abstract="A" * 200,
                        engines=["semantic scholar"])
    unique, removed = dedupe_batch([a, b])
    assert removed == 1 and len(unique) == 1
    assert unique[0].doi == "10.1000/abc" and len(unique[0].abstract) == 200


def test_merge_metadata_only_fills_gaps():
    updates = merge_metadata({"doi": "10.1000/a", "abstract": "short", "engines": ["arxiv"]},
                             {"doi": "10.1000/b", "pdf_url": "https://x/y.pdf", "abstract": "long " * 30,
                              "engines": ["crossref"]})
    assert "doi" not in updates
    assert updates["pdf_url"] == "https://x/y.pdf"
    assert updates["engines"] == ["arxiv", "crossref"]
    assert updates["abstract"].startswith("long")


def test_preprint_and_published_version_are_one_work():
    preprint = {"title": "Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph",
                "doi": "10.48550/arxiv.2412.15268"}
    published = {"title": "Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph",
                 "doi": "10.18653/v1/2025.findings-acl.1270", "venue": "Findings of ACL 2025"}
    assert find_duplicate(published, [preprint]) is preprint
    updates = merge_metadata(preprint, published)
    assert updates["doi"] == "10.18653/v1/2025.findings-acl.1270" and updates["venue"] == "Findings of ACL 2025"
    # Two different published DOIs still mean two different papers.
    other = {**published, "doi": "10.1145/9999999.1111111"}
    assert find_duplicate(other, [published]) is None
