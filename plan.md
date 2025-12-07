this project looks at whether you can use the counterfactual importance value from the
paper thought anchors as a target for a linear probe

this repo has been given a reward_hacking_prompts.json which contains prompts known to elicit reward hacking from gpt-oss-20b

gpt-oss-20b is running on vllm, port 8000

the intent of this experiment is
    1. create 1 rollout for each of the prompts
    2. break that rollout by sentence (regex on ". ") to get a list of sentences
    3. for each sentence, do 50 rollouts from that sentence to get P(reward hacks | seq)
    4. the end result should be a json which is like a list showing sentence by sentence what the P(reward hacks | seq up to know) is, such that I can come back later and get activations + labels easily.

I want code to be neat and modular. High quality and reproducible. There should be
reusable components in utils, for example for querying the api, doing chat template stuff.