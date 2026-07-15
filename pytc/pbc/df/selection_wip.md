# Experimental selection-mode completion map

Current branch: `28a5f57` has the explicit experimental modes but must not be
run because validation work is included in each build.

Remaining amendment before review:

1. Make `cached_full` select once from cached AO and pass cached AO blocks to
   eta construction, so it performs one full-grid AO evaluation total.
2. Split panel measurement into `panel_dense` and `panel_oracle`; both consume
   the same deterministic candidate identities, and each builds and times one
   selector only. Move their equality check to an untimed test.
3. Add `selection_mode` to `ISDFDF` and return mode, candidate rule/count and
   indices, pivots, realized rank, AO calls and AO grid points split between
   selection-only and through-eta totals, plus cache and panel bytes.
4. Add tests for A equality and B dense/oracle equality outside build timing.

No production default changes and no executable comparison are authorized.
