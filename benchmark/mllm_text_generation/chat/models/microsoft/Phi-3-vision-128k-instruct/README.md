- Added `chat_template`, which is mostly a hack to make the benchmark work. It assumes that:
  - There is only one image
  - The messages array is length 2, first with role "system" and then "user"
- Chat format according the the HuggingFace Hub README:
   ```
   <|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n
   ```

### Chat template

```jinja
{{ '<|user|>\n<|image_1|>\n' + messages[0]['content'] + ' ' + messages[1]['content'] + '<|end|>\n<|assistant|>\n' }}
```
