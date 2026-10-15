# The final JevAgentHarness: both provisioning rows (exploratory, descriptive; no test)

| row | workers | pairs | success llm_only / harness | Δ success pt | $ / ep. llm_only / harness (q) | saving | wall s / ep. llm_only / harness | Δ wall s | model-call s / ep. llm_only / harness | S2 calls / ep. |
|---|---|---|---|---|---|---|---|---|---|---|
| shared GPU (about 4 harness episodes per GPU) | 8 | 200 | 99.5 / 100.0 | +0.5 | 0.0434 / 0.0078 (0.00192) | 82.1 % | 8.70 / 22.39 | +13.69 | 8.43 / 22.05 | 7.52 / 3.05 |
| dedicated GPU (at most 1 harness episode per GPU) | 1 | 200 | 99.5 / 100.0 | +0.5 | 0.0431 / 0.0102 (0.00438) | 76.3 % | 13.59 / 9.59 | -4.00 | 11.68 / 9.04 | 7.53 / 3.04 |

Intervals (paired task bootstrap, 95 %) are in each row's summary file. Host: one NVIDIA A10G (g5.4xlarge); q = $1.624 / h × the row's own tariff window / episodes.
