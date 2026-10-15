# Serving benchmark

device/dtype: cuda / bfloat16; torch 2.14.1+cu130, transformers 5.18.0; repeats 3, steps/episode 6

- `Qwen/Qwen3-0.6B`: cache DynamicCache {'DynamicLayer': 28}, croppable=True, 112.0 KB/token
- `openbmb/MiniCPM5-1B`: cache DynamicCache {'DynamicLayer': 24}, croppable=True, 24.0 KB/token
- `Qwen/Qwen3.5-0.8B`: cache DynamicCache {'LinearAttentionLayer': 18, 'DynamicLayer': 6}, croppable=False, 397.9 KB/token

| model | source | state tok | q | B | mode | ms/decision p50 | p95 | ms/request p50 | prefilled tok/dec | stateless tok/dec | reuse x | speedup vs stateless | phases ms/request |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | synthetic:1000 | 1103 | 1 | 1 | stateless | 31.2 | 51.2 | 31.2 | 891 | 891 | 1.0 | 1.00 | - |
| Qwen3-0.6B | synthetic:1000 | 1103 | 1 | 1 | prefix | 31.0 | 51.6 | 31.0 | 891 | 891 | 1.0 | 1.01 | - |
| Qwen3-0.6B | synthetic:1000 | 1103 | 1 | 1 | session | 37.4 | 37.8 | 37.4 | 471 | 891 | 1.9 | 0.83 | encode=1.1 lookup=0.3 prefill=34.3 store=0.4 suffix=0.2 |
| Qwen3-0.6B | synthetic:1000 | 1100 | 3 | 1 | stateless | 35.6 | 73.0 | 106.7 | 751 | 751 | 1.0 | 1.00 | - |
| Qwen3-0.6B | synthetic:1000 | 1100 | 3 | 1 | prefix | 24.9 | 33.4 | 74.6 | 329 | 751 | 2.3 | 1.43 | encode=2.8 prefill=33.5 suffix=44.3 |
| Qwen3-0.6B | synthetic:1000 | 1100 | 3 | 1 | session | 25.6 | 29.5 | 76.8 | 189 | 751 | 4.0 | 1.39 | encode=2.8 lookup=0.3 prefill=34.2 store=0.2 suffix=41.0 |
| Qwen3-0.6B | synthetic:4000 | 4066 | 1 | 1 | stateless | 75.5 | 205.5 | 75.5 | 2374 | 2374 | 1.0 | 1.00 | - |
| Qwen3-0.6B | synthetic:4000 | 4066 | 1 | 1 | prefix | 75.1 | 207.8 | 75.1 | 2374 | 2374 | 1.0 | 1.01 | - |
| Qwen3-0.6B | synthetic:4000 | 4066 | 1 | 1 | session | 67.3 | 96.8 | 67.3 | 964 | 2374 | 2.5 | 1.12 | encode=2.4 lookup=0.3 prefill=64.0 store=0.4 suffix=0.2 |
| Qwen3-0.6B | synthetic:4000 | 4069 | 3 | 1 | stateless | 123.3 | 409.0 | 370.0 | 2235 | 2235 | 1.0 | 1.00 | - |
| Qwen3-0.6B | synthetic:4000 | 4069 | 3 | 1 | prefix | 45.7 | 118.5 | 137.0 | 824 | 2235 | 2.7 | 2.70 | encode=6.8 prefill=98.4 suffix=78.0 |
| Qwen3-0.6B | synthetic:4000 | 4069 | 3 | 1 | session | 40.3 | 66.4 | 120.9 | 354 | 2235 | 6.3 | 3.06 | encode=6.9 lookup=0.3 prefill=48.7 store=0.2 suffix=76.0 |
| Qwen3-0.6B | synthetic:16000 | 15947 | 1 | 1 | stateless | 385.4 | 1339.0 | 385.4 | 8310 | 8310 | 1.0 | 1.00 | - |
| Qwen3-0.6B | synthetic:16000 | 15947 | 1 | 1 | prefix | 386.7 | 1345.6 | 386.7 | 8310 | 8310 | 1.0 | 1.00 | - |
| Qwen3-0.6B | synthetic:16000 | 15947 | 1 | 1 | session | 401.3 | 767.5 | 401.3 | 2945 | 8310 | 2.8 | 0.96 | encode=7.8 lookup=0.5 prefill=426.0 store=0.5 suffix=0.2 |
| Qwen3-0.6B | synthetic:16000 | 15969 | 3 | 1 | stateless | 846.7 | 3675.0 | 2540.2 | 8186 | 8186 | 1.0 | 1.00 | - |
| Qwen3-0.6B | synthetic:16000 | 15969 | 3 | 1 | prefix | 220.1 | 887.4 | 660.4 | 2807 | 8186 | 2.9 | 3.85 | encode=22.2 prefill=904.8 suffix=200.1 |
| Qwen3-0.6B | synthetic:16000 | 15969 | 3 | 1 | session | 182.2 | 374.1 | 546.7 | 1015 | 8186 | 8.1 | 4.65 | encode=23.0 lookup=0.5 prefill=396.2 store=0.3 suffix=201.1 |
| MiniCPM5-1B | synthetic:1000 | 1065 | 1 | 1 | stateless | 37.9 | 61.9 | 37.9 | 882 | 882 | 1.0 | 1.00 | - |
| MiniCPM5-1B | synthetic:1000 | 1065 | 1 | 1 | prefix | 37.7 | 64.7 | 37.7 | 882 | 882 | 1.0 | 1.01 | - |
| MiniCPM5-1B | synthetic:1000 | 1065 | 1 | 1 | session | 31.3 | 34.2 | 31.3 | 474 | 882 | 1.9 | 1.21 | encode=1.5 lookup=0.3 prefill=27.9 store=0.3 suffix=0.2 |
| MiniCPM5-1B | synthetic:1000 | 1065 | 3 | 1 | stateless | 38.5 | 73.9 | 115.4 | 737 | 737 | 1.0 | 1.00 | - |
| MiniCPM5-1B | synthetic:1000 | 1065 | 3 | 1 | prefix | 27.2 | 35.0 | 81.5 | 327 | 737 | 2.3 | 1.42 | encode=3.7 prefill=31.0 suffix=50.1 |
| MiniCPM5-1B | synthetic:1000 | 1065 | 3 | 1 | session | 25.6 | 27.9 | 76.9 | 191 | 737 | 3.9 | 1.50 | encode=3.8 lookup=0.3 prefill=24.5 store=0.2 suffix=47.0 |
| MiniCPM5-1B | synthetic:4000 | 4064 | 1 | 1 | stateless | 96.0 | 217.7 | 96.0 | 2382 | 2382 | 1.0 | 1.00 | - |
| MiniCPM5-1B | synthetic:4000 | 4064 | 1 | 1 | prefix | 95.0 | 217.8 | 95.0 | 2382 | 2382 | 1.0 | 1.01 | - |
| MiniCPM5-1B | synthetic:4000 | 4064 | 1 | 1 | session | 65.0 | 91.6 | 65.0 | 974 | 2382 | 2.4 | 1.48 | encode=3.7 lookup=0.3 prefill=59.8 store=0.4 suffix=0.2 |
| MiniCPM5-1B | synthetic:4000 | 4064 | 3 | 1 | stateless | 122.3 | 374.0 | 367.0 | 2236 | 2236 | 1.0 | 1.00 | - |
| MiniCPM5-1B | synthetic:4000 | 4064 | 3 | 1 | prefix | 49.3 | 109.0 | 147.9 | 827 | 2236 | 2.7 | 2.48 | encode=10.1 prefill=99.9 suffix=71.1 |
| MiniCPM5-1B | synthetic:4000 | 4064 | 3 | 1 | session | 42.3 | 60.5 | 126.8 | 357 | 2236 | 6.3 | 2.90 | encode=10.2 lookup=0.3 prefill=48.8 store=0.2 suffix=69.7 |
| MiniCPM5-1B | synthetic:16000 | 16062 | 1 | 1 | stateless | 397.0 | 1248.6 | 397.0 | 8381 | 8381 | 1.0 | 1.00 | - |
| MiniCPM5-1B | synthetic:16000 | 16062 | 1 | 1 | prefix | 401.2 | 1260.4 | 401.2 | 8381 | 8381 | 1.0 | 0.99 | - |
| MiniCPM5-1B | synthetic:16000 | 16062 | 1 | 1 | session | 373.1 | 714.5 | 373.1 | 2974 | 8381 | 2.8 | 1.06 | encode=14.0 lookup=0.5 prefill=389.5 store=0.5 suffix=0.2 |
| MiniCPM5-1B | synthetic:16000 | 16063 | 3 | 1 | stateless | 761.7 | 3200.5 | 2285.0 | 8236 | 8236 | 1.0 | 1.00 | - |
| MiniCPM5-1B | synthetic:16000 | 16063 | 3 | 1 | prefix | 194.9 | 770.5 | 584.8 | 2826 | 8236 | 2.9 | 3.91 | encode=39.1 prefill=799.0 suffix=144.0 |
| MiniCPM5-1B | synthetic:16000 | 16063 | 3 | 1 | session | 160.3 | 314.0 | 480.9 | 1024 | 8236 | 8.0 | 4.75 | encode=39.1 lookup=0.5 prefill=360.2 store=0.3 suffix=142.1 |
| Qwen3.5-0.8B | synthetic:1000 | 1111 | 1 | 1 | stateless | 110.7 | 160.8 | 110.7 | 900 | 900 | 1.0 | 1.00 | - |
| Qwen3.5-0.8B | synthetic:1000 | 1111 | 1 | 1 | prefix | 110.6 | 160.6 | 110.6 | 900 | 900 | 1.0 | 1.00 | - |
| Qwen3.5-0.8B | synthetic:1000 | 1111 | 1 | 1 | session | 178.5 | 244.0 | 178.5 | 525 | 900 | 1.7 | 0.62 | encode=1.2 lookup=0.5 prefill=148.1 store=1.5 suffix=36.0 |
| Qwen3.5-0.8B | synthetic:1000 | 1108 | 3 | 1 | stateless | 77.9 | 151.2 | 233.7 | 760 | 760 | 1.0 | 1.00 | - |
| Qwen3.5-0.8B | synthetic:1000 | 1108 | 3 | 1 | prefix | 57.7 | 73.5 | 173.0 | 333 | 760 | 2.3 | 1.35 | encode=3.0 prefill=98.7 suffix=77.5 |
| Qwen3.5-0.8B | synthetic:1000 | 1108 | 3 | 1 | session | 67.2 | 88.7 | 201.5 | 208 | 760 | 3.7 | 1.16 | encode=3.0 lookup=0.5 prefill=127.5 store=1.5 suffix=77.4 |
| Qwen3.5-0.8B | synthetic:4000 | 4074 | 1 | 1 | stateless | 208.1 | 504.1 | 208.1 | 2384 | 2384 | 1.0 | 1.00 | - |
| Qwen3.5-0.8B | synthetic:4000 | 4074 | 1 | 1 | prefix | 207.9 | 505.6 | 207.9 | 2384 | 2384 | 1.0 | 1.00 | - |
| Qwen3.5-0.8B | synthetic:4000 | 4074 | 1 | 1 | session | 225.7 | 335.8 | 225.7 | 1118 | 2384 | 2.1 | 0.92 | encode=2.5 lookup=0.5 prefill=193.1 store=1.6 suffix=36.0 |
| Qwen3.5-0.8B | synthetic:4000 | 4077 | 3 | 1 | stateless | 225.1 | 551.6 | 675.4 | 2244 | 2244 | 1.0 | 1.00 | - |
| Qwen3.5-0.8B | synthetic:4000 | 4077 | 3 | 1 | prefix | 92.1 | 202.9 | 276.4 | 828 | 2244 | 2.7 | 2.44 | encode=7.0 prefill=250.7 suffix=83.5 |
| Qwen3.5-0.8B | synthetic:4000 | 4077 | 3 | 1 | session | 86.6 | 121.0 | 259.8 | 406 | 2244 | 5.5 | 2.60 | encode=6.9 lookup=0.5 prefill=173.0 store=1.6 suffix=80.6 |
| Qwen3.5-0.8B | synthetic:16000 | 15955 | 1 | 1 | stateless | 848.0 | 2375.5 | 848.0 | 8320 | 8320 | 1.0 | 1.00 | - |
| Qwen3.5-0.8B | synthetic:16000 | 15955 | 1 | 1 | prefix | 850.1 | 2382.4 | 850.1 | 8320 | 8320 | 1.0 | 1.00 | - |
| Qwen3.5-0.8B | synthetic:16000 | 15955 | 1 | 1 | session | 540.9 | 2279.8 | 540.9 | 5625 | 8320 | 1.5 | 1.57 | encode=7.9 lookup=0.5 prefill=771.4 store=2.0 suffix=36.2 |
| Qwen3.5-0.8B | synthetic:16000 | 15977 | 3 | 1 | stateless | 888.1 | 2598.5 | 2664.2 | 8195 | 8195 | 1.0 | 1.00 | - |
| Qwen3.5-0.8B | synthetic:16000 | 15977 | 3 | 1 | prefix | 293.3 | 797.1 | 880.0 | 2811 | 8195 | 2.9 | 3.03 | encode=22.9 prefill=1037.5 suffix=114.4 |
| Qwen3.5-0.8B | synthetic:16000 | 15977 | 3 | 1 | session | 204.2 | 808.0 | 612.6 | 1912 | 8195 | 4.3 | 4.35 | encode=23.0 lookup=0.5 prefill=758.0 store=2.1 suffix=110.0 |
