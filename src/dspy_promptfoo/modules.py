"""
Example DSPy Modules for Promptfoo Integration
Demonstrates various DSPy capabilities including signatures, optimization, and different module types.
"""

import dspy
from typing import List, Dict, Any


# Simple Q&A Module
class SimpleQA(dspy.Module):
    """Basic question-answering module"""
    
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.generate_answer(question=question)


# Chain of Thought Module
class ExplainedQA(dspy.Module):
    """Question-answering with reasoning"""
    
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate_answer(question=question)


# Multi-hop QA Module
class MultiHopQA(dspy.Module):
    """Multi-step reasoning module"""
    
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        # Generate search query
        query = self.generate_query(question=question).search_query
        
        # Simulate retrieval (in real usage, this would call a retriever)
        context = f"Retrieved context for query: {query}"
        
        # Generate answer using context
        return self.generate_answer(context=context, question=question)


# Classification Module
class TextClassifier(dspy.Module):
    """Text classification module"""
    
    def __init__(self, categories: List[str]):
        super().__init__()
        self.categories = categories
        categories_str = ", ".join(categories)
        self.classify = dspy.Predict(f"text -> category: Literal[{categories_str}]")
    
    def forward(self, text):
        return self.classify(text=text)


# Summarization Module
class Summarizer(dspy.Module):
    """Text summarization module"""
    
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("text -> summary")
    
    def forward(self, text):
        return self.summarize(text=text)


# Code Generation Module
class CodeGenerator(dspy.Module):
    """Code generation with explanation"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            "description -> code: str, explanation: str"
        )
    
    def forward(self, description):
        return self.generate(description=description)


# Custom Signature Examples
class EmailDrafter(dspy.Module):
    """Email drafting with tone control"""
    
    def __init__(self):
        super().__init__()
        self.draft = dspy.Predict(
            "recipient: str, subject: str, tone: str, key_points: str -> email: str"
        )
    
    def forward(self, recipient, subject, tone, key_points):
        return self.draft(
            recipient=recipient,
            subject=subject,
            tone=tone,
            key_points=key_points
        )


# RAG Module Example
class RAGModule(dspy.Module):
    """Retrieval-Augmented Generation module"""
    
    def __init__(self, passages_per_hop=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        # Retrieve relevant passages
        passages = self.retrieve(question).passages
        
        # Format context from passages
        context = "\n".join(passages[:3])  # Use top 3 passages
        
        # Generate answer
        return self.generate_answer(context=context, question=question)


# Example training data for optimization
def get_example_trainset():
    """Get example training data for DSPy optimization"""
    
    # Q&A examples
    qa_examples = [
        dspy.Example(
            question="What is the capital of France?",
            answer="Paris"
        ).with_inputs("question"),
        
        dspy.Example(
            question="Who wrote Romeo and Juliet?",
            answer="William Shakespeare"
        ).with_inputs("question"),
        
        dspy.Example(
            question="What is the speed of light?",
            answer="299,792,458 meters per second"
        ).with_inputs("question"),
    ]
    
    # Classification examples
    classification_examples = [
        dspy.Example(
            text="I love this product! Best purchase ever.",
            category="positive"
        ).with_inputs("text"),
        
        dspy.Example(
            text="This is terrible. Complete waste of money.",
            category="negative"
        ).with_inputs("text"),
        
        dspy.Example(
            text="It's okay, nothing special.",
            category="neutral"
        ).with_inputs("text"),
    ]
    
    # Code generation examples
    code_examples = [
        dspy.Example(
            description="Write a Python function to calculate factorial",
            code="""def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)""",
            explanation="This recursive function calculates the factorial of a number."
        ).with_inputs("description"),
    ]
    
    return {
        "qa": qa_examples,
        "classification": classification_examples,
        "code": code_examples
    }


# Optimization example
def optimize_module(module: dspy.Module, trainset: List[dspy.Example], metric=None):
    """Optimize a DSPy module using BootstrapFewShot"""
    
    if metric is None:
        # Default metric: check if output contains expected answer
        def metric(gold, pred, trace=None):
            # Get the first output field
            output_fields = [f for f in gold.keys() if f not in gold.inputs().keys()]
            if output_fields:
                field = output_fields[0]
                if hasattr(pred, field) and hasattr(gold, field):
                    gold_value = str(getattr(gold, field)).lower()
                    pred_value = str(getattr(pred, field)).lower()
                    return gold_value in pred_value or pred_value in gold_value
            return True
    
    # Use BootstrapFewShot optimizer
    from dspy.teleprompt import BootstrapFewShot
    
    optimizer = BootstrapFewShot(metric=metric)
    optimized = optimizer.compile(module, trainset=trainset)
    
    return optimized


# Example usage
if __name__ == "__main__":
    # Configure DSPy
    import os
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    dspy.settings.configure(lm=lm)
    
    # Test simple QA
    qa = SimpleQA()
    result = qa(question="What is DSPy?")
    print(f"Simple QA: {result.answer}")
    
    # Test Chain of Thought
    cot = ExplainedQA()
    result = cot(question="Why is the sky blue?")
    print(f"\nChain of Thought: {result.answer}")
    print(f"Reasoning: {result.reasoning}")
    
    # Test classifier
    classifier = TextClassifier(categories=["positive", "negative", "neutral"])
    result = classifier(text="This framework is amazing!")
    print(f"\nClassification: {result.category}")
    
    # Test code generation
    codegen = CodeGenerator()
    result = codegen(description="Write a function to reverse a string")
    print(f"\nCode Generation:")
    print(f"Code: {result.code}")
    print(f"Explanation: {result.explanation}")
