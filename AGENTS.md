# AGENTS.md

## GROUND RULES - NEVER BREAK

- Before implementing a major feature, create a plan following the plan guidelines below.
- Start writing code only AFTER the plan has been approved, unless it is a quick bugfix or the user explicitly asks to skip planning. A quick bugfix is typically anything that touches only one file and involves changing fewer than 40 lines of code.
- While implementing features, touch the fewest files possible and write the least code required to solve the problem. 
- Strongly apply the DRY principle to reuse abstractions and functions.
- Do NOT create ANY new SHARED abstractions on the fly - make sure it is detailed in the plan and the user accepts it first.
- During implementation, remove all dead code as it gets created.
- Create strong tests to verify the implementations. Ensure 100% code coverage for touched code (non-negotiable).
- `uv` is the package manager we use for everything related to Python. Do not do anything Python related without using `uv`.
- While reviewing code, follow the review guidelines below.

---

## Plan Guidelines

Before designing any new feature, it must be clearly planned out. The plan must highlight every aspect of the design, foreseeable issues, concrete tradeoffs and give the user an opinionated recommendation, and ask for their input before finalizing the plan. The priority order for tradeoffs is - Correctness > Maintainability > Performance.

**Engineering preferences:**

* **DRY is important**—flag repetition aggressively.
* **Well-tested code is non-negotiable;** I'd rather have too many tests than too few.
* Features should be **"engineered enough"** — not under-engineered (fragile, hacky) nor over-engineered (premature abstraction, unnecessary complexity).
* **Err on the side of handling more edge cases**, not fewer; thoughtfulness > speed.
* **Bias toward explicit** over clever.

---

## Review Guidelines

### 1. Code Review

**Evaluate:**

* Code organization and module structure.
* **DRY violations**—be aggressive here.
* Error handling patterns and missing edge cases (call these out explicitly).
* Technical debt hotspots.
* Areas that are over-engineered or under-engineered relative to my preferences.

### 2. Test review

**Evaluate:**

* Test coverage gaps (unit, integration, e2e).
* Test quality and assertion strength.
* Missing edge case coverage—be thorough.
* Untested failure modes and error paths.

### 3. Performance review

**Evaluate:**

* Slow or high-complexity code paths.
* Memory-usage concerns.
* Caching opportunities.

### For each actionable issue you find

* Describe the problem concretely, with file and line references.
* Present 2–3 options, including "do nothing" where that's reasonable.
* For each option, specify: implementation effort, risk, impact on other code, and maintenance burden.
* Give me your recommended option and why, mapped to my preferences above.
* Then explicitly ask whether I agree or want to choose a different direction before proceeding.
