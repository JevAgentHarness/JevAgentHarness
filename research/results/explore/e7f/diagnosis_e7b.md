## Diagnosis: `jevagent_best_rl` in `e7b_hard (a copy outside the checkout)`

- episodes: 200 complete of 200 tasks
- thresholds ran with: tool_select cascade tau 0.4999999999978404; tool_params cascade tau 0.9999998807907104; reflection trigger tau 0.5
- tool_select: 2881 cascade decisions; accepted / deferred as the threshold says: 2881 (of which 0 deferred because the answer was OTHER); violations 0
- tool_params: 4751 cascade decisions; accepted / deferred as the threshold says: 4751 (of which 997 deferred because the answer was OTHER); violations 0
- System-1 endpoint per episode: {"backend": "jevagentrl-4b", "base_url": "http://127.0.0.1:8081", "extra_body": {}, "model": "jevagentrl-4b", "provider": "typesafe"} x 200
- System-1 calls by backend / model: jevagentrl-4b / jevagentrl-4b x 10313
- write states (gold next tool cancel_order / ship_order): 957; both writes offered 957; the policy sentence in the task 957; the order identified from the current observation 950, and of those the customer's tier and the amount both in the state 950; longest state 5102 characters (harness cap 12000; states at the cap 0)

| tier \| amount vs. the tier's limit \| gold | n | model wrong | accepted wrong |
|---|---|---|---|
| basic | above limit | gold cancel_order | 207 | 10 (4.8 %) | 10 |
| basic | above limit | gold ship_order | 2 | 1 (50.0 %) | 1 |
| basic | at or below limit | gold ship_order | 92 | 44 (47.8 %) | 44 |
| gold | gold tier | gold ship_order | 332 | 6 (1.8 %) | 6 |
| silver | above limit | gold cancel_order | 115 | 20 (17.4 %) | 20 |
| silver | at or below limit | gold ship_order | 202 | 97 (48.0 %) | 97 |

Cancel rate by tier and amount band ($150 bands; the rule cancels basic above $300, silver above $600, never gold):

| tier \| amount band ($) | n | model says cancel % | rule cancels % |
|---|---|---|---|
| basic | 0-150 | 45 | 35.6 | 0.0 |
| basic | 150-300 | 47 | 59.6 | 0.0 |
| basic | 300-450 | 56 | 94.6 | 98.2 |
| basic | 450-600 | 65 | 93.8 | 100.0 |
| basic | 600-750 | 41 | 92.7 | 97.6 |
| basic | 750-900 | 47 | 97.9 | 100.0 |
| gold | 0-150 | 53 | 0.0 | 0.0 |
| gold | 150-300 | 50 | 0.0 | 0.0 |
| gold | 300-450 | 71 | 0.0 | 0.0 |
| gold | 450-600 | 50 | 0.0 | 0.0 |
| gold | 600-750 | 54 | 0.0 | 0.0 |
| gold | 750-900 | 54 | 0.0 | 0.0 |
| silver | 0-150 | 43 | 2.3 | 0.0 |
| silver | 150-300 | 51 | 35.3 | 0.0 |
| silver | 300-450 | 61 | 72.1 | 0.0 |
| silver | 450-600 | 47 | 68.1 | 0.0 |
| silver | 600-750 | 60 | 83.3 | 100.0 |
| silver | 750-900 | 55 | 81.8 | 100.0 |

Failed episodes by type, with the decision that made the last step:

- 139 x strict_stop: wrong tool cancel_order (gold ship_order), tool chosen by the model
- 30 x strict_stop: wrong tool ship_order (gold cancel_order), tool chosen by the model
- 24 x premature_finish: finish chosen by the model
- 5 x wrong_total: last step finish chosen by the model
- 1 x no_answer: last step find_user chosen by the model
- early finishes, by the number of writes done before the finish: 6 writes: 1, 7 writes: 1, 8 writes: 4, 9 writes: 2, 10 writes: 10, 11 writes: 6

