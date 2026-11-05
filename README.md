# briefagent

A small Python CLI agent that reads a company's public website and writes a
one-page sourced brief: what they sell, who they sell to, and one outreach
hook. Every claim carries the URL the agent actually fetched to support it.

No agent framework. Plain Python plus one provider SDK. The point of the
project is not the agent, it is the eval: measuring whether letting a model
choose which pages to read beats not letting it.

**It does not.** Against a fixed keyword crawl with the same page budget, the
agent used 2.4x the tokens and verified 11.9 fewer points of its citations,
with a confidence interval excluding zero. See [Results](#results).

## How it works

The loop is smaller than people expect. The model is handed two tools as JSON
schemas, and the program obeys whichever one it calls.

1. Send the conversation to the model, with `fetch_url` and `submit_brief`
   attached.
2. The model replies with text blocks, tool-call blocks, or both.
3. If it called `fetch_url`, run the real fetch and append the cleaned page
   text to the conversation as a new message. Go to step 1.
4. If it called `submit_brief`, validate the claims and stop.
5. If a limit hits first, force a submit with whatever it has.

Three details do most of the work:

**The conversation is the memory.** Every fetched page is appended as a
message. Turn four can reason about turn one's page only because that text is
still in the message list. There is no store. This is also why page text is
capped: six uncapped pages would blow both the context and the budget.

**Stopping is a tool call, not a parse.** Many naive agents stop when the
model's prose "looks final", which is a regex guess. Here the brief arrives as
structured JSON through `submit_brief`, which means the eval reads claims as
data instead of regex-parsing markdown, and every system produces identically
shaped output.

**Citations are enforced, not requested.** The prompt tells the model to cite
only pages it fetched. After submission, `validate_claims` checks every
`source_url` against the set of URLs actually fetched and deletes anything
else, recording it under "Dropped claims" in the brief. It also checks whether
`cited_excerpt` appears verbatim in that page and sets `excerpt_found`, which
catches invented quotes with no human in the loop.

## Architecture

```
            company URL
                 |
                 v
        +------------------+
        |    agent.py      |  the loop, tool schemas, claim validation
        +------------------+
           |            |
           v            v
      llm.py        tools.py
   one model call   fetch_url
   + retries        + SSRF guard
   + provider       + robots.txt
     dispatch       + rate limit
           |            |
           +-----+------+
                 v
            trace.py            one JSONL line per step
                 |
                 v
           render.py            claims JSON -> brief.md
                 |
                 v
        brief.md + trace.jsonl + claims.json
```

| File | Responsibility |
|---|---|
| `briefagent/agent.py` | The loop, the two tool schemas, the system prompt, claim validation, CLI |
| `briefagent/llm.py` | `llm_call()`, the single swappable model call. Dispatches to a provider |
| `briefagent/gemini.py` | Gemini adapter. Translates messages, tools, schemas and tool choice both ways |
| `briefagent/openai_compat.py` | One adapter for Groq, Cerebras, OpenRouter, GitHub Models and Ollama |
| `briefagent/tools.py` | `fetch_url`, HTML cleaning, SSRF guard, robots.txt, rate limiter |
| `briefagent/retry.py` | Exponential backoff with full jitter, shared by the model call and the fetch |
| `briefagent/trace.py` | JSONL trace writer and running token/cost totals |
| `briefagent/render.py` | Claims JSON to markdown |
| `briefagent/config.py` | Every tunable value, all overridable by env var |
| `briefagent/__init__.py` | Version string, used by `--version` and `--check` |
| `eval/` | Runner, the two baselines, claim extraction, scoring |

Adding a third provider touched `llm.py` and one new file. `agent.py`,
`tools.py`, `trace.py`, `render.py` and every eval script were untouched.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env     # then fill in your key
```

Pick a provider. The agent loop is identical either way; only `llm.py` and the
adapter know which one is running.

**Groq, Cerebras, OpenRouter, GitHub Models, local Ollama.** All speak the
OpenAI chat completions format, so one adapter covers them. Groq's free tier
has no credit card requirement and is what the published results were run on.

```bash
export BRIEFAGENT_PROVIDER=groq        # or cerebras, openrouter, github, ollama
export GROQ_API_KEY=...
export BRIEFAGENT_MODEL=openai/gpt-oss-120b
```

Each has a default model, a default pacing interval and the env var it reads
its key from, all in `PROVIDERS` at the top of `openai_compat.py`. For anything
else that speaks the same format, use `openai_compat` with
`BRIEFAGENT_BASE_URL` and `BRIEFAGENT_API_KEY`.

Ollama runs locally, needs no key, and is free without limit, but needs a model
that supports tool calling such as `llama3.1` or `qwen2.5`, and will be slow on
a laptop.

**Gemini, free tier.** No credit card. Get a key at
[aistudio.google.com](https://aistudio.google.com).

```bash
export BRIEFAGENT_PROVIDER=gemini
export GEMINI_API_KEY=...
export BRIEFAGENT_MODEL=gemini-3.8-flash
```

Gemini free tier quota is per **project**, not per API key. A second key in the
same project adds nothing. If AI Studio asks for billing, you are creating the
key inside a project that already has a billing account linked. Create the key
in a new project to stay on the free tier.

**Anthropic, paid.** Prepaid credits, no free tier.

```bash
export BRIEFAGENT_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export BRIEFAGENT_MODEL=claude-sonnet-5-5
```

Then:

```bash
python -m briefagent.agent https://example.com --out runs
```

Writes `runs/example-com/brief.md`, `claims.json` and `trace.jsonl`.

### Check the setup before you run anything

```bash
python -m briefagent.agent --check
```

Verifies the build version, the provider, the SDK, that a key is set, the
worst-case request against the token limit, and then makes **one** real call
with a tool attached to confirm the model actually returns a tool call. It
warns rather than passing when that call is slow. Every check corresponds to a
failure hit during development.

This matters because the test suite is deliberately offline. A green `pytest`
proves the logic works and says nothing about whether the provider is wired up.
`--check` is the first thing that talks to the network.

```bash
python -m briefagent.agent --list-models   # what this key can actually use
python -m briefagent.agent --version       # confirm which build you have
python eval/run.py --estimate              # token cost of a full eval
python eval/run.py --show-errors           # what went wrong last run
```

### Cost estimation is opt-in on purpose

Prices are not hardcoded, because a hardcoded price table goes stale and a
README full of invented dollar figures is worse than none. Set these to the
current published price per million tokens before any run whose cost you intend
to report:

```bash
export BRIEFAGENT_PRICE_IN_PER_MTOK=3.00
export BRIEFAGENT_PRICE_OUT_PER_MTOK=15.00
```

Unset, every cost reads 0 and the trace marks `cost_priced: false`, which is
accurate on a free tier rather than flattering.

### Useful knobs

| Env var | Default | What it does |
|---|---|---|
| `BRIEFAGENT_PROVIDER` | `anthropic` | `anthropic`, `gemini`, `groq`, `cerebras`, `openrouter`, `github`, `ollama`, `openai_compat` |
| `BRIEFAGENT_MODEL` | per provider | Model name. Confirm with `--list-models` |
| `BRIEFAGENT_MAX_TOOL_CALLS` | 6 | Hard cap on fetches per company |
| `BRIEFAGENT_MAX_STEPS` | 10 | Hard cap on model turns |
| `BRIEFAGENT_FETCH_CHAR_CAP` | 6000 | Page text given to the model |
| `BRIEFAGENT_LLM_MIN_INTERVAL` | per provider | Seconds between model calls |
| `BRIEFAGENT_LLM_TIMEOUT` | 90 | Seconds before one model call gives up |
| `BRIEFAGENT_TPM_LIMIT` | per provider | Tokens per minute, 0 disables |
| `BRIEFAGENT_TPD_LIMIT` | unset | Daily token budget, used by `--estimate` |
| `BRIEFAGENT_THINKING_BUDGET` | 0 | Gemini thinking tokens. 0 off, -1 automatic |
| `BRIEFAGENT_BUDGET_MAX_TOKENS` | 60000 | Abort threshold per run |
| `BRIEFAGENT_BUDGET_MAX_USD` | 0.50 | Abort threshold, only when prices are set |
| `BRIEFAGENT_RETRY_ATTEMPTS` | 6 | Attempts before giving up |
| `BRIEFAGENT_RATE_LIMIT_SECONDS` | 1.0 | Seconds between outbound page fetches |
| `BRIEFAGENT_RESPECT_ROBOTS` | 1 | Set to 0 to ignore robots.txt |

## Safety

| Control | Behaviour |
|---|---|
| Scheme allowlist | Only `http` and `https`. `file://`, `data:`, `gopher://` and the rest are refused |
| SSRF guard | Hostnames are resolved and every returned address is checked. Private, loopback, link-local, reserved, multicast and unspecified ranges are refused, including `169.254.169.254` |
| Redirect validation | Redirects are followed one hop at a time, by hand, with the full check repeated at each hop. Letting the HTTP client follow them automatically would let a public URL bounce to `127.0.0.1` and skip the guard entirely |
| robots.txt | Checked per origin and cached. A denial is returned to the model as a tool error so it picks a different page, and shows up in the trace |
| Rate limit | One request per second across the process |
| Scope | Company-level public pages. URLs matching personal profile patterns are refused, matched on registrable domain and path rather than substring, so `darwinbox.com` is not mistaken for `x.com` |
| Side effects | None. The tool reads pages and writes local files. It never sends anything anywhere |

## Output

`brief.md`, one page, every claim followed by its source link, plus a "Dropped
claims" section when the model cited something it never fetched.

`trace.jsonl`, one line per step:

```json
{"type":"step","step":2,"action":"fetch_url","url":"https://example.com/about",
 "http_status":200,"chars":3184,"latency_ms":412,"cost_usd":0.0,"ok":true,"error":null}
{"type":"step","step":3,"action":"model_call","tokens_in":4210,"tokens_out":286,
 "latency_ms":2914,"cost_usd":0.0,"reasoning":"The about page named...","error":null}
```

The last line is always a `run_summary` with totals, which is what the scoring
script reads. Committed runs for all 30 systems are in `eval/results/`.

## Eval method

The question being tested: **does letting the model choose which pages to read
produce better sourced claims than a fixed crawl?**

Three systems, identical claim rules, identical `submit_brief` schema,
identical validation. The only difference is page selection.

| System | Page selection | Why it is here |
|---|---|---|
| `agent` | Model picks each page, up to `MAX_TOOL_CALLS` fetches, adapting to what it read | The thing being tested |
| `baseline_fixed` | Homepage plus up to 3 links matched by a keyword rule, fetched blindly, one model call | The real control. Comparable page budget, no decisions |
| `baseline_home` | Homepage text only, one model call, no tools | The floor |

`baseline_fixed` is the comparison that matters. `baseline_home` can only ever
cite one URL, so it loses on source coverage by construction. Beating it proves
nothing.

### Procedure

```bash
python eval/run.py --estimate   # will this fit in your daily quota?
python eval/run.py              # 10 companies x 3 systems = 30 runs
python eval/extract.py          # writes eval/claims.csv
python eval/score.py            # tier 1 metrics, no labelling needed

# optional tier 2: label the `supported` column by hand, then re-run score.py
```

Re-running resumes by default, keeping completed runs and retrying only
failures. `--fresh` forces a full redo. Re-running `extract.py` preserves
labels you have already entered.

`eval/companies.txt` holds the 10 companies, chosen by fetching each one first
and spreading them across homepage text volume (1,347 characters at the thin
end up to the cap at the rich end). The file comments record the measurements
and the four bot-blocked sites that were deliberately excluded.

### The harness refuses broken runs

`eval/run.py` ends with a run quality table and exits non-zero if any system
produced usable briefs on fewer than half the companies. `eval/extract.py`
refuses to build the labelling CSV in that case.

This is not defensive padding. On an early run a token limit killed 20 of 30
runs, every exception was caught and filed as an empty brief, and the harness
printed "done". Two of three systems had produced nothing, and the next step
would have been an afternoon of hand labelling a comparison that did not exist.
A crashed run is now recorded as `crashed`, separately from a brief genuinely
truncated by a budget. The first is a failed measurement, the second is a
result.

### Two tiers of metric

**Tier 1, automatic.** Computed from the run itself, no human judgement, no
labelling. Available the moment `eval/run.py` finishes. These measure
*grounding*: does the quoted text actually appear on the page the claim cites,
was that page actually fetched, and how many distinct sources did the system
use.

- citation verification rate, with a 95% bootstrap CI resampled by company
- fabricated citation rate (claims citing a page never fetched, removed by the validator)
- distinct sources per brief, claims per brief, completion, crashes
- steps, fetches, tokens, latency, cost

The CI resamples companies, not claims. Claims from one company are correlated
(a thin website drags down several at once), so resampling claims would give an
interval that is too narrow.

**Tier 2, hand labelled.** Needs a human to open each source and decide whether
it supports the claim. This measures *correctness*, which no automatic check
can. It is the stronger metric and it is optional: `score.py` reports tier 1
whether or not any labels exist. **Not run for the results below.**

Tier 2 also reports an exact McNemar test on per-company all-claims-supported,
computed from the binomial directly with no scipy. At 10 companies that test
has almost no power: five discordant pairs cannot reach p < 0.05 even on a
clean sweep. `score.py` prints the smallest split that would have been
significant for whatever n you got, so a null cannot be quietly reported as a
finding.

Two more guards in the scoring:

- A brief with zero claims counts as a failure, not a free win. A system that
  says nothing cannot win on accuracy by default.
- Claims per brief sits next to the rate, and the script warns when one system
  produces far fewer claims than another. A system at 90% on 1 claim per brief
  is not beating one at 85% on 6. It is answering less of the question.

## Results

Run config, 2026-11-05:

| | |
|---|---|
| Provider | Groq, free tier |
| Model | `openai/gpt-oss-120b` |
| `BRIEFAGENT_FETCH_CHAR_CAP` | 2500 |
| `BRIEFAGENT_MAX_TOOL_CALLS` | 4 |
| Companies | 10 B2B SaaS |
| Completion | 30/30 runs, 0 crashes, 190 claims |

Tier 1, automatic:

| System | Citations verified | 95% CI | Fabricated cites | Sources/brief | Claims/brief |
|---|---|---|---|---|---|
| `agent` | 69.4% (43/62) | [54%, 83%] | 0.0% (0/62) | 2.0 | 6.2 |
| `baseline_fixed` | 81.2% (52/64) | [68%, 93%] | 0.0% (0/64) | 3.3 | 6.4 |
| `baseline_home` | 84.4% (54/64) | [73%, 94%] | 0.0% (0/64) | 1.0 | 6.4 |

| System | Steps | Fetches | Tokens in | Tokens out | Latency |
|---|---|---|---|---|---|
| `agent` | 8.1 | 3.3 | 6,888 | 1,758 | 223s |
| `baseline_fixed` | 5.0 | 4.0 | 2,909 | 1,555 | 89s |
| `baseline_home` | 2.0 | 1.0 | 1,228 | 1,235 | 55s |

| Comparison | Difference | 95% CI | Significant |
|---|---|---|---|
| agent vs `baseline_fixed` | -11.9 points | [-22.7, -1.0] | Yes, the agent is worse |
| agent vs `baseline_home` | -15.0 points | [-35.7, +3.2] | No, interval contains zero |

Tier 2, hand labelled: not run. `eval/claims.csv` has 190 unlabelled rows.

### What the result says

**The agent lost.** Against a fixed keyword crawl with a comparable page budget
it used 2.4x the input tokens and 2.5x the wall time, and its citations
verified 11.9 points less often. The confidence interval excludes zero, so this
is a real difference at n=10 rather than noise.

Against the homepage-only baseline the gap is larger but the interval contains
zero, so nothing is established there.

This is the comparison most agent demos skip. Without `baseline_fixed`, the
agent's 69.4% would have read as a result rather than a loss.

Every system completed every brief, and none fabricated a citation: the
validator removed every claim pointing at an unfetched page before scoring.

### Caveats that scope it

**The agent was budget constrained and the baselines were not.** Free tier
quota forced `MAX_TOOL_CALLS=4`, below the design value of 6, and 4 of 10 agent
runs hit that cap (retool, postman, ramp, snowflake). The finding is that
model-driven page selection lost *under a 4-fetch budget*, not that it loses in
general.

**A hypothesis, untested.** The agent cited 2.0 distinct sources per brief
against the crawl's 3.3, despite a comparable page budget. One explanation
worth testing is context position: the crawl receives all four pages in a
single message, while the agent receives them across separate turns, so earlier
pages sit further back in context when it writes its quotes. That would predict
the gap shrinks if the agent re-summarises pages as it goes.

**Grounding is not correctness.** Citation verification checks whether the
quoted text appears on the cited page. It does not check whether the claim is
true or useful. A system can cite perfectly and still write a worthless brief.

## Tests

```bash
python -m pytest
```

143 tests, fully offline. No API key, no network, no fixtures downloaded. CI
runs them on every push across Python 3.11, 3.12, 3.13 and 3.14, and fails the
build if an em dash appears in any tracked file.

Covered: the loop stops at max steps, the fetch budget is a hard cap, the
budget cap aborts cleanly into a valid incomplete brief, retries fire on
transient failures and not on genuinely malformed requests, backoff grows
exponentially and is capped, the SSRF blocklist catches every private range
including redirect chains into one, lookalike domains are not mistaken for
social profiles, robots.txt denial blocks a fetch, rate and token limiters
sleep, claims citing unfetched pages are dropped, invented excerpts are
flagged, trace lines carry every field the eval reads, and the harness refuses
to report a run where a system produced nothing.

Both provider adapters have their own translation tests, because a provider
adapter does not crash when it is wrong. It quietly sends a malformed schema
and gets back a model that never calls a tool.

## Limitations

**The eval is underpowered and the sample is not random.** Ten companies chosen
by hand is a convenience sample. The result generalises to mid-market B2B SaaS
companies with reachable English or European marketing sites, and not further.
Spreading a convenience sample across text volume does not make it
representative.

**One run per system per company.** Model output varies between runs, so
run-to-run variance is uncontrolled and is being read as system difference.
Three runs per cell with a variance estimate would be the honest version and
costs three times the quota.

**The sample excludes sites that block bots.** ServiceNow, Freshworks and
Personio returned 403 or 429, and SAP disallows the path in robots.txt. They
were dropped because an empty brief from every system carries no information
and wastes one of only ten slots. Defensible for statistical power,
indefensible for external validity: the result says nothing about how either
system handles a site it cannot read.

**Tier 2 is unrun, and would need blinding.** Whoever labels also built the
agent and can often tell which system wrote a claim from its shape. Doing it
properly means shuffling rows, hiding the system column, and ideally a second
labeller with an agreement score. The shuffle is a two-line script; the second
labeller is not.

**Text only, no JavaScript.** Pages rendered client-side come back nearly
empty. Modern marketing sites are often exactly this, which biases the sample
toward server-rendered ones.

**The character cap truncates.** At `FETCH_CHAR_CAP=2500` every system sees the
same reduced text, so the comparison is fair, but briefs are thinner than a
paid tier would produce and the numbers are not comparable to a run at a
different cap.

**The outreach hook is the weakest section.** "Specific and recent" is not
something a validator can check. A hook can be perfectly sourced and useless.

**No deduplication across claims.** Two claims saying the same thing in
different words both count, inflating claims per brief without adding
information.

**Results do not transfer across providers.** A run on Gemini Flash and a run
on Claude Sonnet are different experiments. Record the provider and model next
to the numbers, and never compare an agent run on one against a baseline run on
another. The eval controls page selection, not the model.

## What building this actually taught me

Five failure modes, all found by running the thing rather than reading about
it. None of them were logic bugs; every one was invisible to the obvious check.

**Offline tests prove nothing about integration.** The suite is deliberately
offline so CI needs no key. A green `pytest` therefore passed happily on a
build whose provider wiring did not exist. `--check` is the answer: it is the
first thing that talks to the network.

**A green check can hide an unusable system.** A preflight call once took 164
seconds and reported "Setup looks good". `--check` now projects the eval
runtime from the observed latency and warns instead of passing.

**Token limits are not request limits.** The agent resends every fetched page
on every turn, so context grows linearly and a late request can exceed an
entire minute's allowance on its own. Request pacing cannot fix that;
`openai_compat.py` tracks tokens in a sliding 60 second window and `--check`
refuses to start when the worst case will not fit.

**Failed requests still cost quota.** A 400 or a 429 consumes the token budget
exactly like a success, so recording usage only on a successful response makes
every failure invisible to the throttle, which then sends the next request into
a limit it has already blown. This was the single biggest cause of cascading
rate limit failures during development. Usage is now recorded before sending
and topped up afterwards.

**A 400 can be a bad sample rather than a bad request.** `openai/gpt-oss-20b`
produced three distinct tool-calling protocol violations: arguments that were
not valid JSON, calling a tool never offered, and ignoring a forced tool
choice. All arrive as HTTP 400, which is normally not retryable, but these are
resampling problems and a retry usually fixes them. The 120B model produced the
same failures at a much lower rate. Model choice affected completion rate far
more than it affected brief quality.

One more that is not a bug: **a full eval is most of a free tier's daily
budget.** Groq allows 200,000 tokens per day per model and one clean run costs
175,000 to 250,000. It fits once per day. Re-running on failure is the
expensive mistake, since each attempt consumes quota the successful attempt
then does not have. `python eval/run.py --estimate` exists for that reason.