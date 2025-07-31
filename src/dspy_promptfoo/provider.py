"""
DSPy Provider for Promptfoo
This provider integrates DSPy's programmatic prompt optimization with Promptfoo's evaluation framework.
"""

import os
import json
import dspy
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DSPyProvider:
    """Custom Promptfoo provider that wraps DSPy modules"""
    
    def __init__(self):
        # Configure DSPy with OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize DSPy with OpenAI
        self.lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
        dspy.settings.configure(lm=self.lm, track_usage=True)
        
        # Cache for compiled modules
        self.compiled_modules = {}
    
    def _get_or_create_module(self, config: Dict[str, Any]) -> dspy.Module:
        """Get or create a DSPy module based on configuration"""
        module_type = config.get("module_type", "predict")
        signature = config.get("signature", "question -> answer")
        
        # Create unique key for caching
        cache_key = f"{module_type}_{signature}"
        
        if cache_key in self.compiled_modules:
            return self.compiled_modules[cache_key]
        
        # Create the appropriate DSPy module
        if module_type == "predict":
            module = dspy.Predict(signature)
        elif module_type == "chain_of_thought":
            module = dspy.ChainOfThought(signature)
        elif module_type == "program_of_thought":
            module = dspy.ProgramOfThought(signature)
        elif module_type == "react":
            # ReAct requires tools
            tools = config.get("tools", [])
            if not tools:
                # Provide a simple default tool
                def dummy_tool(x: str) -> str:
                    return f"Processed: {x}"
                tools = [dummy_tool]
            module = dspy.ReAct(signature, tools=tools)
        else:
            # Default to Predict
            module = dspy.Predict(signature)
        
        # Apply optimization if requested
        if config.get("optimize", False):
            module = self._optimize_module(module, config)
        
        self.compiled_modules[cache_key] = module
        return module
    
    def _optimize_module(self, module: dspy.Module, config: Dict[str, Any]) -> dspy.Module:
        """Optimize a DSPy module using provided examples"""
        optimizer_type = config.get("optimizer", "BootstrapFewShot")
        examples = config.get("examples", [])
        
        if not examples:
            # Return unoptimized module if no examples provided
            return module
        
        # Convert examples to DSPy format
        trainset = []
        for ex in examples:
            dspy_ex = dspy.Example(**ex).with_inputs(*ex.get("inputs", ["question"]))
            trainset.append(dspy_ex)
        
        # Create metric function
        def metric(gold, pred, trace=None):
            # Simple metric: check if answer is present
            if hasattr(pred, "answer") and hasattr(gold, "answer"):
                return gold.answer.lower() in pred.answer.lower()
            return True
        
        # Apply optimizer
        if optimizer_type == "BootstrapFewShot":
            from dspy.teleprompt import BootstrapFewShot
            optimizer = BootstrapFewShot(metric=metric)
            compiled = optimizer.compile(module, trainset=trainset)
        elif optimizer_type == "BootstrapFewShotWithRandomSearch":
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch
            optimizer = BootstrapFewShotWithRandomSearch(metric=metric, num_candidate_programs=3)
            compiled = optimizer.compile(module, trainset=trainset)
        else:
            # Default: return unoptimized
            compiled = module
        
        return compiled
    
    def _extract_output(self, result: Any, config: Dict[str, Any]) -> str:
        """Extract output from DSPy result based on configuration"""
        output_field = config.get("output_field", "answer")
        
        # Handle different result types
        if hasattr(result, output_field):
            return str(getattr(result, output_field))
        elif isinstance(result, dict) and output_field in result:
            return str(result[output_field])
        else:
            # Try to convert to string
            return str(result)
    
    def _get_usage_stats(self, result: Any) -> Dict[str, Any]:
        """Extract usage statistics from DSPy result"""
        usage = {}
        
        # Try to get LM usage if available
        if hasattr(result, "get_lm_usage"):
            lm_usage = result.get_lm_usage()
            if lm_usage:
                total_tokens = 0
                prompt_tokens = 0
                completion_tokens = 0
                
                for entry in lm_usage:
                    if isinstance(entry, dict):
                        usage_data = entry.get("usage", {})
                        total_tokens += usage_data.get("total_tokens", 0)
                        prompt_tokens += usage_data.get("prompt_tokens", 0)
                        completion_tokens += usage_data.get("completion_tokens", 0)
                
                if total_tokens > 0:
                    usage = {
                        "total": total_tokens,
                        "prompt": prompt_tokens,
                        "completion": completion_tokens
                    }
        
        return usage


def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for Promptfoo provider
    
    Args:
        prompt: The prompt text
        options: Provider configuration from promptfooconfig.yaml
        context: Additional context including test variables
    
    Returns:
        Dictionary with output and optional metadata
    """
    try:
        # Initialize provider
        provider = DSPyProvider()
        
        # Get configuration
        config = options.get("config", {})
        
        # Get or create DSPy module
        module = provider._get_or_create_module(config)
        
        # Prepare inputs
        vars = context.get("vars", {})
        
        # Build kwargs for module call
        kwargs = {}
        
        # Handle different input formats
        if "{{" in prompt:
            # Prompt has template variables, use them
            for key, value in vars.items():
                kwargs[key] = value
        else:
            # Use prompt as question by default
            kwargs["question"] = prompt
            # Add any additional vars
            kwargs.update(vars)
        
        # Call the DSPy module
        result = module(**kwargs)
        
        # Extract output
        output = provider._extract_output(result, config)
        
        # Build response
        response = {
            "output": output
        }
        
        # Add usage stats if available
        usage = provider._get_usage_stats(result)
        if usage:
            response["tokenUsage"] = usage
        
        # Add metadata
        response["metadata"] = {
            "module_type": config.get("module_type", "predict"),
            "signature": config.get("signature", "question -> answer"),
            "optimized": config.get("optimize", False),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add full result for debugging if requested
        if config.get("debug", False):
            response["debug"] = {
                "full_result": str(result),
                "type": type(result).__name__
            }
        
        return response
        
    except Exception as e:
        return {
            "error": f"DSPy Provider Error: {str(e)}",
            "output": ""  # Provide empty output on error
        }


# Example usage for testing
if __name__ == "__main__":
    # Test the provider
    test_prompt = "What is the capital of France?"
    test_options = {
        "config": {
            "module_type": "chain_of_thought",
            "signature": "question -> answer",
            "debug": True
        }
    }
    test_context = {"vars": {}}
    
    result = call_api(test_prompt, test_options, test_context)
    print(json.dumps(result, indent=2))
