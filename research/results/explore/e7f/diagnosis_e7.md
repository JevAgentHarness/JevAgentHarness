## Diagnosis: `jevagent_best_rl` in `e7_hard (a copy outside the checkout)`

- episodes: 200 complete of 200 tasks
- thresholds ran with: tool_select cascade tau 0.4999999999978404; tool_params cascade tau 0.9999998807907104; reflection trigger tau 0.5
- tool_select: 2670 cascade decisions; accepted / deferred as the threshold says: 2670 (of which 0 deferred because the answer was OTHER); violations 0
- tool_params: 4370 cascade decisions; accepted / deferred as the threshold says: 4370 (of which 853 deferred because the answer was OTHER); violations 0
- System-1 endpoint per episode: {"backend": "jevagentrl-4b", "base_url": "http://127.0.0.1:8081", "extra_body": {}, "model": "jevagentrl-4b", "provider": "typesafe"} x 200
- System-1 calls by backend / model: jevagentrl-4b / jevagentrl-4b x 9510
- write states (gold next tool cancel_order / ship_order): 877; both writes offered 877; the policy sentence in the task 877; the order identified from the current observation 876, and of those the customer's tier and the amount both in the state 876; longest state 5097 characters (harness cap 12000; states at the cap 0)

| tier \| amount vs. the tier's limit \| gold | n | model wrong | accepted wrong |
|---|---|---|---|
| basic | above limit | gold cancel_order | 210 | 7 (3.3 %) | 7 |
| basic | at or below limit | gold ship_order | 102 | 52 (51.0 %) | 52 |
| gold | gold tier | gold ship_order | 273 | 9 (3.3 %) | 8 |
| silver | above limit | gold cancel_order | 102 | 10 (9.8 %) | 10 |
| silver | at or below limit | gold ship_order | 189 | 83 (43.9 %) | 83 |

Cancel rate by tier and amount band ($150 bands; the rule cancels basic above $300, silver above $600, never gold):

| tier \| amount band ($) | n | model says cancel % | rule cancels % |
|---|---|---|---|
| basic | 0-150 | 45 | 35.6 | 0.0 |
| basic | 150-300 | 57 | 63.2 | 0.0 |
| basic | 300-450 | 45 | 88.9 | 100.0 |
| basic | 450-600 | 57 | 96.5 | 100.0 |
| basic | 600-750 | 57 | 100.0 | 100.0 |
| basic | 750-900 | 51 | 100.0 | 100.0 |
| gold | 0-150 | 38 | 0.0 | 0.0 |
| gold | 150-300 | 55 | 0.0 | 0.0 |
| gold | 300-450 | 44 | 0.0 | 0.0 |
| gold | 450-600 | 47 | 0.0 | 0.0 |
| gold | 600-750 | 48 | 4.2 | 0.0 |
| gold | 750-900 | 41 | 0.0 | 0.0 |
| silver | 0-150 | 39 | 2.6 | 0.0 |
| silver | 150-300 | 61 | 24.6 | 0.0 |
| silver | 300-450 | 45 | 77.8 | 0.0 |
| silver | 450-600 | 44 | 65.9 | 0.0 |
| silver | 600-750 | 39 | 84.6 | 100.0 |
| silver | 750-900 | 63 | 93.7 | 100.0 |

Failed episodes by type, with the decision that made the last step:

- 134 x strict_stop: wrong tool cancel_order (gold ship_order), tool chosen by the model
- 37 x premature_finish: finish chosen by the model
- 17 x strict_stop: wrong tool ship_order (gold cancel_order), tool chosen by the model
- 5 x wrong_total: last step finish chosen by the model
- 1 x strict_stop: ship_order with a wrong argument, tool chosen by the model
- early finishes, by the number of writes done before the finish: 4 writes: 3, 5 writes: 14, 6 writes: 4, 8 writes: 6, 9 writes: 10

