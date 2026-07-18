"""sec_edgar.py — CIK/accession parsing, Item-code extraction, and the material-item gate that
decides whether a filing is worth the extra fetch. No network calls (those are exercised manually
against live EDGAR — see the docstring in sec_edgar.py); this locks down the pure-parsing pieces
that the 2026-07-18 SEC EDGAR fix depends on."""
import sec_edgar


def test_extract_cik_accession_from_index_url():
    url = "https://www.sec.gov/Archives/edgar/data/1141197/000165495426006724/0001654954-26-006724-index.htm"
    cik, accession = sec_edgar.extract_cik_accession(url)
    assert cik == 1141197
    assert accession == "000165495426006724"


def test_extract_cik_accession_handles_missing_or_malformed_url():
    assert sec_edgar.extract_cik_accession("") == (None, None)
    assert sec_edgar.extract_cik_accession("not a url") == (None, None)


def test_item_codes_in_summary_parses_multiple_items():
    summary = (" Filed: 2026-07-17 AccNo: 0001654954-26-006724 Size: 191 KB\n"
               "Item 5.02: Departure of Directors or Certain Officers\n"
               "Item 9.01: Financial Statements and Exhibits")
    assert sec_edgar.item_codes_in_summary(summary) == {"5.02", "9.01"}


def test_item_codes_in_summary_empty_on_no_items():
    assert sec_edgar.item_codes_in_summary("") == set()
    assert sec_edgar.item_codes_in_summary("no item codes here") == set()


def test_material_items_gate_fetch_worthy_filing():
    """A material corporate action (executive departure) should trigger the extra fetch."""
    items = sec_edgar.item_codes_in_summary("Item 5.02: Departure of Directors")
    assert items & sec_edgar.MATERIAL_ITEMS


def test_material_items_gate_skips_routine_only_filing():
    """Bylaw amendments / vote results alone are NOT worth the extra SEC request -- this is the
    gate that keeps fetch volume proportional to genuinely interesting filings instead of every
    8-K (see sec_edgar.py docstring)."""
    items = sec_edgar.item_codes_in_summary(
        "Item 5.03: Amendments to Articles of Incorporation\nItem 9.01: Financial Statements and Exhibits")
    assert not (items & sec_edgar.MATERIAL_ITEMS)


def test_strip_html_skip_to_item_removes_cover_page_boilerplate():
    """Every primary 8-K opens with ~2KB of identical SEC cover-page boilerplate before the real
    Item narrative -- verified live 2026-07-18 that a naive small char-budget cutoff landed
    entirely inside that boilerplate. skip_to_item must jump past it."""
    html = ("<html><body>UNITED STATES SECURITIES AND EXCHANGE COMMISSION WASHINGTON D.C. "
            "FORM 8-K CURRENT REPORT registrant name address CIK 0001141197 "
            "Item 5.02 Departure of Directors. On July 15, 2026, Mr. Smith resigned.</body></html>")
    out = sec_edgar._strip_html(html, max_chars=200, skip_to_item=True)
    assert out.startswith("Item 5.02")
    assert "resigned" in out


def test_strip_html_without_skip_to_item_keeps_start_of_document():
    """Exhibits (press releases) have no cover-page header, so they should NOT be skipped."""
    html = "<html><body>Company Announces Record Quarterly Revenue of $500M</body></html>"
    out = sec_edgar._strip_html(html, max_chars=200, skip_to_item=False)
    assert out.startswith("Company Announces")


def test_cik_ticker_map_resolves_known_cik(monkeypatch, tmp_path):
    """Isolated from the real network/cache: point the on-disk cache at a tmp file that already
    has fresh data, so load_cik_ticker_map returns it without any HTTP call."""
    import time
    import json as _json

    cache_file = tmp_path / "sec_cik_tickers.json"
    cache_file.write_text(_json.dumps({
        "0": {"cik_str": 1655210, "ticker": "BYND", "title": "Beyond Meat, Inc."},
    }))
    monkeypatch.setattr(sec_edgar, "CIK_TICKER_CACHE", str(cache_file))
    monkeypatch.setattr(sec_edgar, "_cik_ticker_map", {})
    monkeypatch.setattr(sec_edgar, "_cik_ticker_loaded_at", 0.0)

    import asyncio

    class _ExplodingSession:
        def get(self, *a, **k):
            raise AssertionError("should not hit the network when a fresh cache file exists")

    result = asyncio.run(sec_edgar.load_cik_ticker_map(_ExplodingSession(), "test-agent"))
    assert result == {1655210: "BYND"}
