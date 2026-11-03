import httpx
import pytest

from src.research import content
from src.research.content import fetch_content, parse_html
from src.research.searxng import SearchUnavailable, SearXNGClient
from tests.conftest import make_searx_transport


def make_pdf(text: str) -> bytes:
    import fitz
    doc = fitz.open()
    for chunk in [text[i:i + 1500] for i in range(0, len(text), 1500)]:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 560, 800), chunk, fontsize=8)
    data = doc.tobytes()
    doc.close()
    return data


LANDING = """<html><head>
<meta name="citation_title" content="LLM Test Generation">
<meta name="citation_author" content="Smith, Alice"><meta name="citation_author" content="Lee, Bob">
<meta name="citation_abstract" content="We evaluate large language models for generating unit tests across many projects and report coverage and failure modes in detail for practitioners.">
<meta name="citation_pdf_url" content="/pdfs/paper.pdf">
<meta name="citation_doi" content="10.1145/1.2">
<meta name="citation_publication_date" content="2023/05/02">
<script>var x = "ignore me";</script>
</head><body><nav>Menu</nav><p>Short landing page.</p></body></html>"""

BODY = ("Introduction. Large language models generate tests. " * 30 + "Method. We prompt models. " * 30 +
        "Results. Coverage improved by 12 percent. " * 30 + "Conclusion. Promising. " * 20)


def test_parse_html_extracts_scholarly_meta():
    meta, text = parse_html(LANDING, "https://pub.example.org/article/1")
    assert meta["title"] == "LLM Test Generation"
    assert meta["authors"] == ["Smith, Alice", "Lee, Bob"]
    assert meta["pdf_url"] == "https://pub.example.org/pdfs/paper.pdf"
    assert meta["doi"] == "10.1145/1.2"
    assert "ignore me" not in text and "Menu" not in text


def test_generic_description_not_trusted_on_non_scholarly_pages():
    meta, _ = parse_html('<meta name="description" content="Buy our product now, best prices ever!">')
    assert "abstract" not in meta


def test_fetch_content_follows_citation_pdf():
    pdf = make_pdf(BODY)

    def handler(request):
        if request.url.path.endswith(".pdf"):
            return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})
        return httpx.Response(200, text=LANDING, headers={"content-type": "text/html"})

    res = fetch_content({"title": "LLM Test Generation", "url": "https://pub.example.org/article/1",
                         "abstract": ""}, transport=httpx.MockTransport(handler))
    assert res.status == "full_text" and res.source == "pdf"
    assert "Coverage improved by 12 percent" in res.text
    assert res.metadata["pdf_url"] == "https://pub.example.org/pdfs/paper.pdf"
    assert res.metadata["abstract"].startswith("We evaluate large language models")
    assert res.metadata["year"] == 2023


def test_fetch_content_falls_back_to_abstract():
    res = fetch_content({"title": "T", "url": "https://pub.example.org/x", "abstract": "A" * 300,
                         "arxiv_id": "2305.00001"},
                        transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    assert res.status == "abstract" and res.text == "A" * 300
    assert any("PDF unavailable" in n for n in res.notes)


def test_fetch_content_metadata_only_and_unavailable():
    t = httpx.MockTransport(lambda r: httpx.Response(500))
    assert fetch_content({"title": "Only a title", "url": "https://x.example.org/a"}, transport=t).status == \
        "metadata_only"
    assert fetch_content({"title": "", "url": ""}, transport=t).status == "unavailable"


def test_fetch_blocks_private_addresses():
    called = []

    def handler(request):
        called.append(request.url)
        return httpx.Response(200, text="secret")

    res = fetch_content({"title": "T", "url": "http://localhost:8000/admin", "abstract": ""},
                        transport=httpx.MockTransport(handler))
    assert called == []
    assert any("non-public" in n for n in res.notes)


def test_redirect_to_private_address_is_blocked():
    def handler(request):
        if request.url.host == "pub.example.org":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/internal"})
        return httpx.Response(200, text="internal data")

    res = fetch_content({"title": "T", "url": "https://pub.example.org/a", "abstract": ""},
                        transport=httpx.MockTransport(handler))
    assert res.status == "metadata_only"
    assert any("non-public" in n for n in res.notes)


# ── SearXNG client ──

def test_searxng_search_normalises_results():
    client = SearXNGClient("http://searx.test", transport=make_searx_transport())
    papers = client.search("anything", max_results=10)
    assert len(papers) == 5
    assert papers[0].arxiv_id == "2305.00001"


def test_searxng_passes_configured_categories_and_engines():
    seen = {}

    def handler(request):
        seen.update(dict(request.url.params))
        return httpx.Response(200, json={"results": []})

    SearXNGClient("http://searx.test", categories="science", engines="arxiv,pubmed",
                  transport=httpx.MockTransport(handler)).search("q")
    assert seen["format"] == "json" and seen["categories"] == "science" and seen["engines"] == "arxiv,pubmed"


@pytest.mark.parametrize("transport,match", [
    (make_searx_transport(fail=True), "unavailable"),
    (httpx.MockTransport(lambda r: httpx.Response(403)), "json"),
    (httpx.MockTransport(lambda r: httpx.Response(502)), "HTTP 502"),
    (httpx.MockTransport(lambda r: httpx.Response(200, text="<html>")), "non-JSON"),
])
def test_searxng_errors_raise_search_unavailable(transport, match):
    with pytest.raises(SearchUnavailable, match=match):
        SearXNGClient("http://searx.test", transport=transport).search("q")


def test_searxng_not_configured():
    ok, msg = SearXNGClient("").check()
    assert not ok and "not configured" in msg
