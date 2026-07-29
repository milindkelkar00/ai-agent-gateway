# Day 1 - What LLMs Actually Are

An LLM is a function. You give it a sequence of tokens, which is basically text, and it predicts the most likely next token one at a time.

It does this after being trained on huge amounts of text, adjusting billions of internal numbers until it becomes very good at that prediction.

That is the core idea.

Reasoning, conversation, knowledge, and other impressive behaviors are all emergent outcomes of doing next-token prediction really well.

In short: an LLM is not magic; it is a highly trained system for predicting the next piece of text.

## What do you mean by predicting the next token?

When we are giving the input does it predict what we will write next? Because those are tokens right?

Let's understand 2 things that I have mixed up: Tokenization and next-token prediction.

Tokenization is how token's get chopped up

Next-token prediction is what the model does with those tokens.

### Let's understand tokenization first
"An LLM is a funtion" doesn't get split into single letters or syllables like "a-n-l" or "m-i-s". A tokenizer usually splits on chunks closer to the whole words or common word pieces.
Roughly it looks like - [an] [LL] [M] [ is] [ a] [ funtcion].

Six tokens, not letter-by-letter. Common short words often become their own token. Longer or rarer words often get split into a couple of chunks.

The exact split depends upon the tokenizer, idea is: tokens are model's vocabulary units, think like wordish pieces not letters.

### Let's understand token prediction now
it's not predicting what the user will type next. It's predicting what model itself should output next, one token at a time as it is generating it's own reply.

the flow looks like:
1. You send: "what is the capital of france?" (this becomes a sequence of tokens)
2. The model looks at that whole sequence and asks "given everything so far, what's the single most likely next token to come after this?" -> it predicts "The"
3. Now the sequence is "What is the capital of france" + "The". It asks the same question again -> predicts "capital"
4. Sequence grows again -> predicts "of"
5. -> predicts "Paris"
6. -> predicts a token meaning "end of response", and stops.

Thus for understanding that response has finished it predicts token.

So the model is repeatedly asking one question - "what token comes next?" - but it is asking this about its own answer, being built one token at a time, with your input as the starting context.

## Inference vs Training

The model already learned to predict tokens well *once*, during the training -- that's when its internal weights got tuned on massive amounts of text (expensive done ahead of time, not something you do). 
Everytime you call the API you are doing **inference** - just running that already trained model to generate predictions.
Nothing about the model changes when you use it.

## How do we call an anthropic API and what do we get in return?
```python
import requests
import os

response = requests.post(
    "https://api.anthropic.com/v1/messages",
    header = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    json={
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
    },
)
data = response.json()
print(data)
```

The response looks like -

```json
{
    "id":"msg_01xyz",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [
        {
            "type": "text",
            "text": "The capital of france is Paris."
        }
    ],
    "stope_reason": "end_turn",
    "stop_sequence": null,
    "usage": {
        "input_tokens": 14,
        "output_tokens": 9
    }
}
```

### Things to note

- **messages** is an array - even for a single question, we are sending a list. This is what is later used for managing conversation history.
- **usage.input_tokens/output_tokens** - this is the tokenization from earlier.  every call tells you exactly how many tokens your text became and how many it generated.
- **stop_reason** - "end_turn" - this is the model deciding to emit that "end of response" token - it stopped because it decided it was done, not because of a length limit.
- **max_tokens** - this is a cap on the output, not the input. It says "generate at most this many tokens, then stope no matter what." 1024 is just a random number taken as an example. If the model's natural response would be longer than the cap, then it get's cut off mid-generation - you'll see "stop_reason" - "max_tokens" and the text will just stop, possibly mid-sentence. It's a safety/cost control, not a suggestion.
- **roles** - In the messages array only two roles exist: "user" and "assistant". There's no "system" inside messagees. Instead, a system prompt(instruction on how a model should behave) goes into a seperate top level parameter called "system", outside the messages arrray.
- *Regarding the system prommpt - we tell the model to get into a persona - like doctor, engineer or tester etc*
- **stop_sequence** - currently null but we can pass a list eg. ["\n\n"] telling the model "stop generating the instant you produce this exact text".
- *we would never interact with these parameters while taking to a chatbot. These are only encountered when we are actually building on top of the API.*
- *While talking to a chat bot some one else has already decided the request format for you - you never touch the max tokens or stop sequences or system directly.*

## Statelessness

The API calls to anthropic are stateless - the model doesn't remeber a thing that you told earlier.

But then how does GPT/Claude remember your conversation?

Pure illusion, done entirely on the client side. Everytime we send a new message - the app is silently resending the entire conversation so far as the messages array - every prior user message and every prior assistant reply - plus your one new message.

The model re-reads the whole thing from scratch every single time and it looks like memory.
