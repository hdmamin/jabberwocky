Old prompt format used by jabberwocky v1 where each prompt is defined by a directory (data/{prompt_name}) containing 2 files: prompt.txt and config.yaml. Jabberwocky v2 condenses each of these into a single {prompt_name}.yaml file.

Prompt | Notes
---|---
simplify_ml | Long prompt that tries to summarize paragraphs of machine learning papers in simple language. 
tldr | Simple summarization with Curie. 
shortest | Addition problem consisting of a few characters for when we want to make a real API call but not use many tokens.
short_dates | Date format normalization (another short no input prompt for testing).
how_to | Enumerate ~5 steps for doing an every task.
eli | Explain in plain language to a second grader. More general than "simplify_ml".
