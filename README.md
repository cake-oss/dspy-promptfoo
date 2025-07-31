# Evaluate DSPy prompts using Promptfoo

DSPy treats prompts as learnable parameters that can be automatically optimized based on your specific task and data. Instead of manually iterating through different phrasings and hoping for improvement, DSPy uses algorithms to systematically search for better prompt formulations. This often discovers non-obvious prompt structures that humans wouldn't naturally think to try.

This demo uses Promptfoo to evaluate DSPY-generated prompts against a fixed human-generated prompt, in order to help you determine which DSPy modules and signatures meet your needs. promptfooconfig.yaml contains the sample EDGAR data and tests against which it evaluates the ouput of the model. Adapt promptfooconfig.yaml to your own use case and re-run the demo script.

## Steps

Git clone this repo:
```bash
git clone https://github.com/kflow-ai/dspy-promptfoo-cake.git
cd dspy-promptfoo-cake
```

Create .env and set appropriate values:
```bash
cp .env.example .env && nano .env
```

Run the Demo script, which installs required packages and performs a Promptfoo evaluation on the built-in EDGAR sample using various DSPy providers and direct OpenAI for comparison.
```bash
./demo.sh
```
## Expected output

<img width="1347" height="720" alt="Screenshot 2025-07-30 at 4 37 13 PM" src="https://github.com/user-attachments/assets/62edb5c5-6528-4854-b7e9-58d6fad44875" />
