- In original `chat_template`, removed calls to `.strip()`, because TGI v1.4.2 does not support strip and we're going to pre-strip all user prompts from the benchmarking script.