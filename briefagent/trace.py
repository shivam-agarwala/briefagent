"""
Trace writer.

One JSON object per line, appended as the run happens so a crashed run still
leaves a readable trace. The eval scripts read these files directly, which is
why the field names are fixed and boring.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config


class Tracer:
    """Writes trace.jsonl and keeps running totals for the run."""

    def __init__(self, path: str | Path, company_url: str, system: str = "agent"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.company_url = company_url
        self.system = system

        self.step = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost = 0.0
        self.tool_calls = 0
        self.errors = 0
        self._started = time.perf_counter()

        # Truncate any previous trace for this company so reruns are clean.
        self._fh = self.path.open("w", encoding="utf-8")
        self._write(
            {
                "type": "run_start",
                "system": system,
                "company_url": company_url,
                "model": config.get_model(),
                "ts": time.time(),
            }
        )

    # -- internals ---------------------------------------------------------

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    # -- public API --------------------------------------------------------

    def log_model_call(
        self,
        tokens_in: int,
        tokens_out: int,
        latency_ms: int,
        stop_reason: str | None = None,
        reasoning: str = "",
        error: str | None = None,
    ) -> float:
        """Record one model call. Returns the cost estimate for it."""
        self.step += 1
        cost = config.estimate_cost(tokens_in, tokens_out)

        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_cost += cost
        if error:
            self.errors += 1

        self._write(
            {
                "type": "step",
                "step": self.step,
                "action": "model_call",
                "url": None,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "cost_usd": round(cost, 6),
                "cost_priced": config.prices_configured(),
                "stop_reason": stop_reason,
                # Truncated so traces stay small but still readable when you
                # want to see why the model picked a page.
                "reasoning": (reasoning or "")[:500],
                "error": error,
            }
        )
        return cost

    def log_fetch(
        self,
        url: str,
        latency_ms: int,
        ok: bool,
        status: int | None = None,
        final_url: str | None = None,
        chars: int = 0,
        error: str | None = None,
    ) -> None:
        """Record one fetch_url call."""
        self.step += 1
        self.tool_calls += 1
        if error:
            self.errors += 1

        self._write(
            {
                "type": "step",
                "step": self.step,
                "action": "fetch_url",
                "url": url,
                "final_url": final_url,
                "http_status": status,
                "chars": chars,
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": latency_ms,
                "cost_usd": 0.0,
                "ok": ok,
                "error": error,
            }
        )

    def log_event(self, event: str, **fields) -> None:
        """Record anything else worth seeing, such as a forced stop."""
        self.step += 1
        self._write(
            {"type": "event", "step": self.step, "action": event, **fields}
        )

    def close(self, outcome: str, claims: int = 0, incomplete: bool = False) -> dict:
        """Write the summary line the eval scripts read, and close the file."""
        summary = {
            "type": "run_summary",
            "system": self.system,
            "company_url": self.company_url,
            "outcome": outcome,
            "steps": self.step,
            "tool_calls": self.tool_calls,
            "claims": claims,
            "incomplete": incomplete,
            "errors": self.errors,
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out,
            "cost_usd": round(self.total_cost, 6),
            "cost_priced": config.prices_configured(),
            "wall_ms": int((time.perf_counter() - self._started) * 1000),
        }
        self._write(summary)
        self._fh.close()
        return summary
