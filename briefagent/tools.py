"""
The agent's only real tool: fetch_url.

Phase 2. Four layers of protection, in the order they run:

  1. Scheme check. Only http and https. Blocks file://, gopher://, data: and
     anything else that is not a web page.
  2. SSRF guard. Resolve the hostname and refuse any address in a private,
     loopback, link-local or reserved range. This is the one that matters. A
     model that reads "visit our internal dashboard at 10.0.0.5" and dutifully
     fetches it has turned your agent into a port scanner inside your own
     network. Cloud metadata endpoints like 169.254.169.254 are the classic
     target.
  3. robots.txt. Checked per host, cached, and a denial is returned to the
     model as a tool error so it can pick a different page.
  4. Rate limit. One request per second across the whole process.

Redirects are followed manually, one hop at a time, because every hop gets the
full check again. Following redirects automatically would let a public URL
bounce you straight to 127.0.0.1 and skip step 2 entirely.

fetch_url still never raises. It returns a dict, always.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from . import config
from .retry import RETRYABLE_STATUS, retry_with_backoff

# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

_JUNK_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "svg",
    "iframe",
    "template",
]

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")


def clean_html(html: str, char_cap: int | None = None) -> str:
    """Turn raw HTML into readable plain text, capped in length."""
    char_cap = config.FETCH_CHAR_CAP if char_cap is None else char_cap
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in _JUNK_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = _BLANK_LINES.sub("\n\n", text).strip()

    if len(text) > char_cap:
        text = text[:char_cap] + "\n\n[truncated at %d characters]" % char_cap
    return text


def extract_links(html: str, base_url: str, same_host_only: bool = True) -> list[tuple]:
    """(url, anchor_text) pairs found in the page. Used by the crawl baseline."""
    soup = BeautifulSoup(html, "html.parser")
    base_host = urlparse(base_url).netloc.lower()
    seen, out = set(), []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if same_host_only and parsed.netloc.lower() != base_host:
            continue
        # Drop the fragment so /about and /about#team are not two pages.
        absolute = urlunparse(parsed._replace(fragment=""))
        if absolute in seen:
            continue
        seen.add(absolute)
        out.append((absolute, anchor.get_text(" ", strip=True)))

    return out


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

# Hosts that are never worth a DNS lookup.
_BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}

# Personal rather than company level. The brief is about an organisation.
#
# Matched as (registrable domain, path prefix), NOT as a substring of the whole
# URL. Substring matching looks fine until you notice that "x.com/" is inside
# "darwinbox.com/", "netbox.com/" and "mailbox.com/", and you have silently
# blocked three real companies. Host and path are checked separately.
_PERSONAL_SITES = (
    ("linkedin.com", "/in/"),
    ("twitter.com", "/"),
    ("x.com", "/"),
    ("facebook.com", "/"),
    ("instagram.com", "/"),
    ("tiktok.com", "/"),
)


def _host_matches(hostname: str, domain: str) -> bool:
    """True only for the domain itself or a subdomain of it."""
    return hostname == domain or hostname.endswith("." + domain)


def is_personal_page(hostname: str, path: str) -> bool:
    path = path or "/"
    return any(
        _host_matches(hostname, domain) and path.startswith(prefix)
        for domain, prefix in _PERSONAL_SITES
    )


def resolve_host(hostname: str) -> list[str]:
    """Every IP a hostname resolves to.

    Its own function so tests can replace it without touching real DNS.
    """
    infos = socket.getaddrinfo(hostname, None)
    return sorted({info[4][0] for info in infos})


def is_blocked_ip(address: str) -> bool:
    """True for any address the agent must not talk to."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True  # unparseable means we cannot prove it is safe

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or not ip.is_global
    )


def check_url(url: str, resolver=resolve_host) -> str | None:
    """Return an error string if the URL must not be fetched, else None.

    Called before the first request and again after every redirect hop.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return "unparseable url"

    if parsed.scheme not in ("http", "https"):
        return "blocked scheme: %s" % (parsed.scheme or "none")

    try:
        hostname = (parsed.hostname or "").lower()
    except ValueError:
        return "unparseable host"

    if not hostname:
        return "missing host"
    if hostname in _BLOCKED_HOSTNAMES:
        return "blocked host: %s" % hostname

    if is_personal_page(hostname, parsed.path):
        return "blocked: personal profile page, company pages only"

    # A literal IP in the URL skips DNS but still gets range checked.
    try:
        ipaddress.ip_address(hostname)
        candidates = [hostname]
    except ValueError:
        try:
            candidates = resolver(hostname)
        except Exception as exc:  # noqa: BLE001
            return "dns failure: %s" % exc
        if not candidates:
            return "dns returned no addresses"

    for address in candidates:
        if is_blocked_ip(address):
            return "blocked private or reserved address: %s" % address

    return None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
_last_request_at = 0.0


def rate_limit(sleep=time.sleep, now=time.monotonic) -> None:
    """Block until at least RATE_LIMIT_SECONDS has passed since the last call."""
    global _last_request_at
    with _rate_lock:
        wait = config.RATE_LIMIT_SECONDS - (now() - _last_request_at)
        if wait > 0:
            sleep(wait)
        _last_request_at = now()


def reset_rate_limit() -> None:
    """Used by tests so one test's timing does not slow the next."""
    global _last_request_at
    _last_request_at = 0.0


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------

_robots_cache: dict = {}


def _robots_for(url: str):
    """Parsed robots.txt for this URL's host, or None if there is not one.

    A missing or unreadable robots.txt means no stated rules, so we allow.
    Cached per origin so a six page crawl fetches it once.
    """
    parsed = urlparse(url)
    origin = "%s://%s" % (parsed.scheme, parsed.netloc)
    if origin in _robots_cache:
        return _robots_cache[origin]

    parser = None
    try:
        response = httpx.get(
            origin + "/robots.txt",
            timeout=config.HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": config.USER_AGENT},
            follow_redirects=True,
        )
        if response.status_code == 200:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(response.text.splitlines())
    except Exception:  # noqa: BLE001 - no robots.txt is not an error
        parser = None

    _robots_cache[origin] = parser
    return parser


def robots_allows(url: str) -> bool:
    if not config.RESPECT_ROBOTS:
        return True
    parser = _robots_for(url)
    if parser is None:
        return True
    return parser.can_fetch(config.USER_AGENT, url)


def reset_robots_cache() -> None:
    _robots_cache.clear()


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------


def _blank_result(url: str) -> dict:
    return {
        "ok": False,
        "url": url,
        "final_url": url,
        "text": "",
        "html": "",
        "status": None,
        "error": None,
        "attempts": 0,
    }


def _get_once(url: str) -> httpx.Response:
    """A single HTTP GET with no redirect following.

    Raises on retryable HTTP statuses so retry_with_backoff can see them.
    """
    rate_limit()
    response = httpx.get(
        url,
        follow_redirects=False,
        timeout=config.HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": config.USER_AGENT},
    )
    if response.status_code in RETRYABLE_STATUS:
        raise httpx.HTTPStatusError(
            "retryable status %d" % response.status_code,
            request=response.request,
            response=response,
        )
    return response


def fetch_url(url: str, include_html: bool = False, resolver=resolve_host) -> dict:
    """Fetch one page safely and return cleaned text.

    Returns a dict with ok, url, final_url, text, html, status, error, attempts.
    Never raises.
    """
    result = _blank_result(url)
    current = url
    attempts_total = 0

    for _hop in range(config.MAX_REDIRECTS + 1):
        # Every hop is checked. This is what makes redirect based SSRF fail.
        blocked = check_url(current, resolver=resolver)
        if blocked:
            result["final_url"] = current
            result["error"] = blocked
            return result

        if not robots_allows(current):
            result["final_url"] = current
            result["error"] = "blocked by robots.txt"
            return result

        attempts = {"n": 0}
        target = current

        def _attempt(target=target):
            attempts["n"] += 1
            return _get_once(target)

        try:
            response = retry_with_backoff(_attempt)
        except Exception as exc:  # noqa: BLE001
            attempts_total += attempts["n"]
            result["final_url"] = current
            result["attempts"] = attempts_total
            result["status"] = getattr(
                getattr(exc, "response", None), "status_code", None
            )
            result["error"] = "%s: %s" % (type(exc).__name__, exc)
            return result

        attempts_total += attempts["n"]
        result["status"] = response.status_code
        result["final_url"] = current
        result["attempts"] = attempts_total

        # Follow one redirect hop, then loop back and re-check everything.
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location")
            if not location:
                result["error"] = "redirect with no location header"
                return result
            current = urljoin(current, location)
            continue

        if response.status_code >= 400:
            result["error"] = "http %d" % response.status_code
            return result

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            result["error"] = "unsupported content-type: %s" % (
                content_type or "unknown"
            )
            return result

        result["text"] = clean_html(response.text)
        if include_html:
            result["html"] = response.text
        result["ok"] = True
        return result

    result["error"] = "too many redirects (limit %d)" % config.MAX_REDIRECTS
    return result
