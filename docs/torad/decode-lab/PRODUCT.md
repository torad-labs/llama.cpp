# Decode comparison

<!-- impeccable:product-schema 1 -->

## Platform

Web, local to the llama.cpp checkout. Static HTML/CSS/JavaScript is the implementation choice for this analysis page, not a new project-wide stack.

## Users and purpose

Marcos and the inference team need to compare the ideal four-RTX-PRO-6000 GLM-5.3 Flash decode budget with actual profiler evidence, down to the smallest operation or transfer the evidence supports.

## Evidence and constraints

The existing model covers DP4, IQ3_XXS/Q8 weights, 400k context and MTP2. Existing Nsight SQLite captures cover one active session at 512/4096 context on runtime 82ad56cb8, with adaptive drafts. These are not matched 40-session/400k benchmarks. Unknown bytes and unassigned operations remain unknown. GPU time summed across devices, elapsed time, CPU API time and per-call time remain separate.

## Success

An operator can inspect a target layer, its individual weight/cache/communication substeps, recorded operation/kernel/copy/API timings, and the assumptions and sources behind each. They can filter by device, search, import another local profile, and change ideal transport assumptions without inventing new measurements. No external upload, paid run or inference-engine change is part of this work.
