# Week 4 Assignment Requirements and Project Choices

Page references below use the PDF page/slide number in `Week 4.pdf`, counting the title slide as page 1.

## A. Mandatory instructions from the lecture

The assignment slide requires the following (page 29):

1. Select a dataset from Kaggle. Online Retail and Groceries are examples, not restrictions.
2. Prepare the data as transaction lists.
3. Perform Association Rule Mining with Apriori.
   - Tune `min_support` and `min_confidence`.
   - Calculate and interpret support and confidence.
4. Present a Top-N rules table and a visualization; a network graph is given as an example.
5. Interpret 3-5 key rules with business or other context-specific insights.

The lecture describes the workflow as first finding itemsets that meet minimum support and then generating rules that meet minimum confidence (page 3). It defines:

- `support(X -> Y) = count(X union Y) / number of transactions`: the proportion of transactions containing both sides of the rule (pages 12-13).
- `confidence(X -> Y) = count(X union Y) / count(X)`: among transactions containing `X`, the proportion that also contain `Y` (pages 12-13 and 23).

The numerical thresholds in the worked examples are illustrative, not assignment defaults: support count 2/confidence 100% (page 4), 50%/50% (page 13), support count 2 (pages 14-21), 70% confidence (page 22), and 0.5/0.7 (page 23). The assignment does not prescribe threshold values, a value for N, a specific binning method, or a train/test split.

### Note on the Apriori property

Page 11 states that every subset of a frequent itemset is frequent, which is the standard Apriori property. Its following sentence says, "Any subset of a non-frequent itemset is also non-frequent." This appears to reverse the intended pruning rule. The standard and mathematically valid statement is:

> Any **superset** of an infrequent itemset is infrequent.

Equivalently, every subset of a frequent itemset must be frequent. This project implements the standard Apriori property while treating the page 11 wording as a likely typographical inversion.

## B. User-selected project design choices

The following choices support the solar-photovoltaic study but are not mandated by page 29:

- **Dataset:** use Kaggle's Solar Power Generation Data to connect the assignment with photonics and photovoltaic operation.
- **Transaction unit and aggregation:** treat each plant timestamp as one transaction. Aggregate inverter-level generation records by `DATE_TIME` before joining them to the plant-level weather-sensor record for that timestamp; this avoids repeating one weather observation for every inverter.
- **Discretization:** convert continuous irradiation, temperature, and power measurements into explicit categorical items such as `Irradiation_Low`, `ModuleTemp_High`, and `ACPower_Medium`. Record the binning method and thresholds so the transactions are reproducible.
- **Lift:** include lift as an optional supplementary measure and ranking aid because the lecture's use case reports it (pages 25-26) and the example network uses it (page 28). Lift is not listed as a required calculation on page 29 and must not replace support or confidence.
- **Holdout validation:** use a chronological holdout, if included, to check whether selected rules remain stable on later observations. Derive bin thresholds and tune rule-mining parameters on the development period, then evaluate the fixed rules on the holdout. A holdout is an additional project-quality check, not a lecture requirement.

These choices remain valid only if the final input to Apriori is an auditable list of categorical-item transactions and the reported rules are calculated from the actual data rather than assumed in advance.
