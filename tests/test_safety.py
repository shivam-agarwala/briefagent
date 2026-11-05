"""Tests for the SSRF guard, robots.txt, rate limiting and scope limits."""

from __future__ import annotations

import pytest

from briefagent import config, tools
from briefagent.tools import check_url, clean_html, extract_links, is_blocked_ip

# A resolver that pretends every hostname is a normal public address.
PUBLIC = lambda host: ["93.184.216.34"]  # noqa: E731


def resolves_to(address):
    return lambda host: [address]


# ---------------------------------------------------------------------------
# IP range blocking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback
        "10.0.0.5",  # private class A
        "172.16.4.2",  # private class B
        "192.168.1.1",  # private class C
        "169.254.169.254",  # cloud metadata, the classic SSRF target
        "0.0.0.0",  # unspecified
        "::1",  # IPv6 loopback
        "fd00::1",  # IPv6 unique local
        "fe80::1",  # IPv6 link local
        "224.0.0.1",  # multicast
        "not-an-ip",  # unparseable means unproven, so blocked
    ],
)
def test_private_and_reserved_addresses_are_blocked(address):
    assert is_blocked_ip(address) is True


@pytest.mark.parametrize("address", ["93.184.216.34", "1.1.1.1", "2606:4700::1111"])
def test_public_addresses_are_allowed(address):
    assert is_blocked_ip(address) is False


# ---------------------------------------------------------------------------
# check_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com",
        "data:text/html,hi",
        "javascript:alert(1)",
    ],
)
def test_non_http_schemes_are_blocked(url):
    assert check_url(url, resolver=PUBLIC) is not None


def test_localhost_is_blocked_by_name():
    assert "blocked host" in check_url("http://localhost:8080/admin", resolver=PUBLIC)


def test_literal_private_ip_is_blocked_without_dns():
    def explode(host):
        raise AssertionError("DNS should not be consulted for a literal IP")

    error = check_url("http://169.254.169.254/latest/meta-data/", resolver=explode)
    assert "private or reserved" in error


def test_public_hostname_resolving_to_private_ip_is_blocked():
    """The dangerous case: a normal looking domain with a private A record."""
    error = check_url("https://internal.acme.test/", resolver=resolves_to("10.1.2.3"))
    assert "private or reserved" in error


def test_dns_failure_is_blocked_not_allowed():
    def fails(host):
        raise OSError("name does not resolve")

    assert "dns failure" in check_url("https://nope.test/", resolver=fails)


def test_ordinary_public_url_passes():
    assert check_url("https://example.com/about", resolver=PUBLIC) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://www.linkedin.com/in/someone",
        "https://x.com/someperson",
        "https://twitter.com/someperson",
        "https://instagram.com/someperson",
        "https://www.facebook.com/someperson",
    ],
)
def test_personal_profiles_are_out_of_scope(url):
    assert "personal profile" in check_url(url, resolver=PUBLIC)


@pytest.mark.parametrize(
    "url",
    [
        # Regression. A substring check for "x.com/" blocks every one of these.
        # All three are real companies found while picking the eval sample.
        "https://www.darwinbox.com/",
        "https://netbox.com/",
        "https://mailbox.com/about",
        # A company page on a social site is still a company page.
        "https://www.linkedin.com/company/acme",
        # A lookalike domain must not inherit the real one's rules.
        "https://notlinkedin.com/in/x",
    ],
)
def test_lookalike_domains_are_not_treated_as_personal_profiles(url):
    assert check_url(url, resolver=PUBLIC) is None


def test_personal_check_matches_subdomains_but_not_suffix_collisions():
    from briefagent.tools import is_personal_page

    assert is_personal_page("www.x.com", "/someone") is True
    assert is_personal_page("x.com", "/someone") is True
    assert is_personal_page("darwinbox.com", "/") is False
    assert is_personal_page("myx.com", "/someone") is False


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


def test_redirect_into_a_private_address_is_blocked(monkeypatch):
    """A public URL that 302s to localhost must not be followed.

    This is why redirects are followed by hand instead of by httpx.
    """
    import httpx

    hops = []

    class FakeResponse:
        def __init__(self, status, headers, text="", url=""):
            self.status_code = status
            self.headers = headers
            self.text = text
            self.request = None

    def fake_get(url, **kwargs):
        hops.append(url)
        if url == "https://public.test/":
            return FakeResponse(302, {"location": "http://127.0.0.1:9000/secrets"})
        return FakeResponse(200, {"content-type": "text/html"}, "<p>secrets</p>")

    monkeypatch.setattr(httpx, "get", fake_get)

    result = tools.fetch_url("https://public.test/", resolver=PUBLIC)

    assert result["ok"] is False
    assert "blocked" in result["error"]
    assert "http://127.0.0.1:9000/secrets" not in hops


def test_redirect_loop_terminates(monkeypatch):
    import httpx

    class FakeResponse:
        def __init__(self, status, headers):
            self.status_code = status
            self.headers = headers
            self.text = ""
            self.request = None

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: FakeResponse(302, {"location": "https://a.test/next"}),
    )
    monkeypatch.setattr(config, "MAX_REDIRECTS", 3)

    result = tools.fetch_url("https://a.test/", resolver=PUBLIC)
    assert result["ok"] is False
    assert "too many redirects" in result["error"]


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


def test_robots_denial_blocks_the_fetch(monkeypatch):
    import urllib.robotparser

    monkeypatch.setattr(config, "RESPECT_ROBOTS", True)
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(["User-agent: *", "Disallow: /private"])
    tools._robots_cache["https://acme.test"] = parser

    assert tools.robots_allows("https://acme.test/private/page") is False
    assert tools.robots_allows("https://acme.test/about") is True

    result = tools.fetch_url("https://acme.test/private/page", resolver=PUBLIC)
    assert result["error"] == "blocked by robots.txt"


def test_missing_robots_means_allowed(monkeypatch):
    monkeypatch.setattr(config, "RESPECT_ROBOTS", True)
    tools._robots_cache["https://acme.test"] = None
    assert tools.robots_allows("https://acme.test/anything") is True


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_sleeps_between_requests(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_SECONDS", 1.0)
    tools.reset_rate_limit()

    slept = []
    clock = {"t": 100.0}

    def fake_sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    tools.rate_limit(sleep=fake_sleep, now=lambda: clock["t"])
    tools.rate_limit(sleep=fake_sleep, now=lambda: clock["t"])

    assert len(slept) == 1
    assert slept[0] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# HTML handling
# ---------------------------------------------------------------------------


def test_clean_html_strips_junk_and_collapses_space():
    html = (
        "<html><head><style>b{}</style></head><body><nav>Home About</nav>"
        "<h1>Acme</h1><p>We   sell   payroll.</p><script>evil()</script></body></html>"
    )
    assert clean_html(html) == "Acme\nWe sell payroll."


def test_clean_html_respects_the_character_cap():
    text = clean_html("<p>" + ("x" * 10000) + "</p>", char_cap=100)
    assert text.startswith("x" * 100)
    assert "truncated at 100" in text


def test_extract_links_keeps_same_host_only():
    html = (
        '<a href="/about">About us</a>'
        '<a href="https://acme.test/pricing">Pricing</a>'
        '<a href="https://other.test/x">Other</a>'
        '<a href="#top">Top</a>'
        '<a href="mailto:a@b.c">Mail</a>'
    )
    links = extract_links(html, "https://acme.test/")
    urls = [u for u, _ in links]
    assert urls == ["https://acme.test/about", "https://acme.test/pricing"]
