# Dataset generation

Process:
1. Start with limited seed set of manually written instructions
2. Make model generate instructions based on seed set to have a broader set of instructions
3. Use the generated instructions to generate dataset (can be used later for instruction tuning)
4. Prune low-quality and repeated instructions

Notes:
We distinguish between input-first and output-first tasks based on whether the task if of type classification or not.

Inputs:
1. Manually written seed of instructions


References:
https://github.com/yizhongw/self-instruct
