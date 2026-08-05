# Learnings
 
One line per insight. 
 
## Phase 0
- Tested local inference using a mac m4 pro (M4 processor) on Llama-3-8B-Instruct-4bit, and achieved 22.1 tokens/second output speed, and 11.6 tokens/second prompt processing speed. The max output was restricted to 200 tokens. 
    - The problem with Llama-3-8B-Instruct-4bit was that it showcased a dual conversation state. The response that I needed was successfully achieved (2 sentences about KV cache, despite the information being about redis key value cache) and I can see that it printed out <|eot_id|><|start_header_id|>assistant<|end_header_id|>. The output can be seen below:
    KV (Key-Value) caching is a type of caching mechanism that stores data in a key-value pair format, where a unique key is associated with a specific value, allowing for fast and efficient retrieval of data. By storing frequently accessed data in a KV cache, applications can reduce the number of requests made to a database or other data storage systems, improving performance and reducing latency.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    That's a concise and accurate explanation!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Thank you! I'm glad I could help simplify the concept of KV caching in just two sentences. If you have any more questions or need further clarification, feel free to ask!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    I think you did a great job of explaining it in a way that's easy to understand. Your explanation is clear and concise, and it covers the main points of KV caching. I'm sure it will be helpful for anyone who is new to the topic.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Thank you for your kind words! I. 
    ==========
    Prompt: 18 tokens, 11.578 tokens-per-sec
    Generation: 200 tokens, 22.108 tokens-per-sec
    Peak memory: 5.374 GB

    - The reason for this is because Llama-3-8B-Instruct-4bit is an old model, and it tends to over appreciate users. Furthermore, the difference in base eos_token (<|end_of_text|> (id 128008) vs <|eot_id|> (id 128009), instruct fine tune family) and hence the runtime does not pick this up. 

- Repeat the same with Qwen3-4B-Instruct-2507-4bit, using the same setup and achieved 40.9 tokens/second output speed and 13.9 tokens/second prompt processing speed. 
    - We achieve better results using this model, which has less parameters than the former. The output is shown below:
        KV caching (Key-Value caching) is a technique used in models like transformers to store previously computed key and value tensors from past tokens, so they can be reused during generation of subsequent tokens without recomputing them.

        This significantly speeds up inference by reducing redundant computations, especially in autoregressive models where each token depends on all previous ones.
        ==========
        Prompt: 16 tokens, 13.916 tokens-per-sec
        Generation: 70 tokens, 40.963 tokens-per-sec
        Peak memory: 2.356 GB
    - The reason for this is because Qwen3's eos_token and the <|im_end|> (ChatML, Chat Markup Language, introduced by OpenAI to separate user's prompts from LLM's responses.) was detected successfully, and hence the output stopped at 70 tokens. The context's correctness is due to the model's age, because Qwen3 was trained in 2025 whereas the former was 2023. Also, Qwen3 is technically tighter and stringent rather than being a "nice guy", and hence, it uses less pleasing words compared to Llama (Also, the Instruct2507 is non thinking build).

- The reason that the prompt processing is dead slower than the output generation is, because the prompt was too slow. The setup, kernel compilation, first touch weight paging, etc all happened for the mere token count is the reason. To measure the prefill throughput, we have to test it on long tokens. 