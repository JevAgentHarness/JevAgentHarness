"""JevAgentRL-v2: the System-1 post-training chain on the Miles RL framework (docs/MILES_PORT.md).

One Miles ``Sample`` is one System-1 decision: the rendered chat prompt (``data.render`` + ``decision_model.chat_text``,
byte for byte the served prompt) followed by ONE option-code token; ``Sample.train_metadata`` carries the option-code
ids of the presented order and the stage's target (SFT / OPD) or sampled event and advantage (GRPO). The custom loss
(``loss.py``) reads the logits of the decision position, restricts them to those code ids and computes exactly the v1
objectives (``decision_model.decision_loss``, ``opd.opd_loss``, ``grpo_online.grpo_decision_loss``). Rollouts
(``rollout.py``) run real ``JevAgent`` episodes whose System-1 is v1's ``TorchDecider`` with its forward pass served by
the Miles SGLang engine (``sglang_decider.py``).

Every module here imports Miles lazily, so the package and its tests run without Miles installed.
"""
