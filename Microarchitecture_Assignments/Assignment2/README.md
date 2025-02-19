# Assignment 2: Covert Channel Bit Inference Assignment

## Task Overview
Participants are required to complete the missing logic in the **receiver** function to infer the transmitted bit based on execution time measurements. The sender encodes a bit (`0` or `1`) using different computational workloads, and the receiver must deduce the bit by analyzing execution cycles.

## What Needs to be Implemented?
- Analyze the **cycles_taken** value in the receiver function.
- Implement logic to determine if the transmitted bit was **0 or 1** based on execution time.
- Print the inferred bit alongside the actual transmitted bit for verification.

## Hints
- Longer execution time typically corresponds to a **bit 1** (due to matrix multiplication).
- Shorter execution time typically corresponds to a **bit 0** (due to NOP instructions).
- You may need to empirically determine a **threshold cycle count** for classification.

## Compilation & Execution
### Compile:
```sh
make
```
### Run:
```sh
./covert 1  # Sending bit 1
./covert 0  # Sending bit 0
```

## Expected Output (Once Implemented)
```
Sent Bit: 1 | Received Bit: 1 (Cycles: XXXX)
Sent Bit: 0 | Received Bit: 0 (Cycles: YYYY)
```
