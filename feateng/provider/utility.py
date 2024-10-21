def make_raw_chat_prompt(
    task_prompt: str,
    tokenizer,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": task_prompt}],
        tokenize=False,
    )
