# E4r: replicate and extension of E4 (mini-SWE-agent, SWE-bench Verified)

**exploratory (E4r), descriptive.** Outside every Holm table; revises no registered result. Routing tau = 0.5 (E4's). Success difference with the paired bootstrap (B = 10,000, seed 0) and Tango 95 % intervals; the one-sided non-inferiority test at 3 points (the paper's rule, metrics.ni_certified_p) is reported descriptively. Cost = API $ per instance as logged (DeepSeek `usage.cost` + Jev); time = agent-run wall clock per instance; savings are ratios of means with paired bootstrap 95 % intervals.

## Pooled (primary): 300 distinct held-out instances = E4's 100 (E4's run) + E4r's 200 new ones

n = 300 instances with a record in every arm (0 of 300 listed missing).

| arm | n | resolved | success % | $ / instance | Jev $ / inst. | mean s | median s | steps | fast-share % | limits exceeded | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| always_deep | 300 | 232 | 77.3 | 0.0397 | 0.0000 | 457.3 | 314.6 | 39.2 | 0.0 | 1 | 0 |
| jevagent | 300 | 229 | 76.3 | 0.0355 | 0.0033 | 435.6 | 300.6 | 38.8 | 22.3 | 1 | 0 |
| always_fast | 300 | 227 | 75.7 | 0.0116 | 0.0000 | 329.9 | 227.7 | 47.3 | 100.0 | 0 | 0 |

| vs always_deep | wins / losses | delta pt [bootstrap 95 %] | Tango 95 % | NI p (3 pt) | NI | cost saved % [95 %] | time saved % [95 %] | co-solved: n, median d s |
|---|---|---|---|---|---|---|---|---|
| jevagent | 9 / 12 | -1.0 [-4.0, 2.0] | [-4.3, 2.2] | 0.108 | no | 10.5 [0.4, 19.5] | 4.7 [-2.4, 11.1] | 220, -17.4 |
| always_fast | 10 / 15 | -1.7 [-5.0, 1.7] | [-5.2, 1.7] | 0.229 | no | 70.7 [66.5, 74.3] | 27.9 [21.0, 34.1] | 217, -62.1 |

## Extension: E4r's 200 new held-out instances

n = 200 instances with a record in every arm (0 of 200 listed missing).

| arm | n | resolved | success % | $ / instance | Jev $ / inst. | mean s | median s | steps | fast-share % | limits exceeded | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| always_deep | 200 | 151 | 75.5 | 0.0389 | 0.0000 | 486.2 | 347.0 | 38.7 | 0.0 | 1 | 0 |
| jevagent | 200 | 148 | 74.0 | 0.0370 | 0.0034 | 476.4 | 317.2 | 39.5 | 21.7 | 1 | 0 |
| always_fast | 200 | 149 | 74.5 | 0.0120 | 0.0000 | 358.6 | 243.5 | 47.5 | 100.0 | 0 | 0 |

| vs always_deep | wins / losses | delta pt [bootstrap 95 %] | Tango 95 % | NI p (3 pt) | NI | cost saved % [95 %] | time saved % [95 %] | co-solved: n, median d s |
|---|---|---|---|---|---|---|---|---|
| jevagent | 6 / 9 | -1.5 [-5.0, 2.0] | [-5.8, 2.6] | 0.240 | no | 4.8 [-8.5, 16.9] | 2.0 [-6.9, 10.4] | 142, -13.5 |
| always_fast | 7 / 9 | -1.0 [-5.0, 3.0] | [-5.3, 3.2] | 0.176 | no | 69.1 [63.4, 73.8] | 26.3 [17.4, 34.3] | 142, -70.4 |

## E4: its 100 evaluation instances (E4's run, as reported)

n = 100 instances with a record in every arm (0 of 100 listed missing).

| arm | n | resolved | success % | $ / instance | Jev $ / inst. | mean s | median s | steps | fast-share % | limits exceeded | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| always_deep | 100 | 81 | 81.0 | 0.0411 | 0.0000 | 399.5 | 237.8 | 40.1 | 0.0 | 0 | 0 |
| jevagent | 100 | 81 | 81.0 | 0.0324 | 0.0033 | 354.1 | 219.0 | 37.4 | 23.5 | 0 | 0 |
| always_fast | 100 | 78 | 78.0 | 0.0108 | 0.0000 | 272.5 | 186.7 | 46.9 | 100.0 | 0 | 0 |

| vs always_deep | wins / losses | delta pt [bootstrap 95 %] | Tango 95 % | NI p (3 pt) | NI | cost saved % [95 %] | time saved % [95 %] | co-solved: n, median d s |
|---|---|---|---|---|---|---|---|---|
| jevagent | 3 / 3 | 0.0 [-5.0, 5.0] | [-5.9, 5.9] | 0.135 | no | 21.3 [8.5, 32.5] | 11.3 [2.0, 20.2] | 78, -25.8 |
| always_fast | 3 / 6 | -3.0 [-9.0, 3.0] | [-10.0, 3.4] | 0.571 | no | 73.7 [68.0, 78.0] | 31.8 [21.0, 41.0] | 75, -38.6 |

## Spend (logged)

| item | $ |
|---|---|
| logged_smoke | 0.02 |
| logged_eval | 17.59 |
| logged_total | 17.61 |
| key_usage | 17.61 |

