# Scientific and engineering caveats

## Questions our tests can answer

- Can code produce a next-week forecast without using unseen next-week observations? YES (prequential walk-forward).
- Does the simple 12-week conversion efficiency forecast beat naive last-week persistence here? YES, on both sampled channels.
- Does a more complex nested model improve accuracy? NO: it slightly underperforms the fixed 12-week model on this dataset.
- Will proposed reallocations always pass risk controls? NO: first two aggressive proposals are rejected.

## Questions this dataset cannot answer

- How many **additional** conversions would a changed budget cause? Unidentified from observational platform counts.
- Whether Meta and Google both attribute the same customer conversion? Unknown.
- Which channel has the better marginal revenue per dollar? Not identified by average CPD.
- Whether the published 2019 data are from a genuine enterprise account? Not independently verified.
- Whether the unexecuted Meridian adapter boosts live business outcomes? No evidence; official data are simulated.

## Before live recommendations

Use independent first-party transaction tables to deduplicate conversions and assign margin; run randomized geo holdouts or experiments; fit carryover and saturation response models with controls, then evaluate time-split holdouts, parameter uncertainty, distribution shift and adverse scenarios. Separately maintain model, application and business KPI monitors. Keep spending controls out of LLM instructions and enforce them in executable code.