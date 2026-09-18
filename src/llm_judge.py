"""
LLM-as-Judge Evaluator for the Movie Safety Classifier.
Uses an LLM to score the quality of classifications on multiple dimensions.
"""

import json
from typing import Dict, List
from dataclasses import dataclass
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import GROQ_API_KEY, GROQ_MODEL


@dataclass
class JudgeScore:
    """Represents a single judge score."""
    correctness: int      # 1-5: Is the classification correct?
    reasoning: int        # 1-5: Is the explanation sound?
    safety: int           # 1-5: Does it err on the side of caution?
    clarity: int          # 1-5: Is the response clear?
    overall: float        # Average
    feedback: str         # Text feedback


class LLMJudge:
    """
    Uses an LLM to evaluate the quality of movie safety classifications.
    """

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = GROQ_MODEL

        print("🔄 Initializing LLM-as-Judge...")
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.0,  # Deterministic scoring
            max_tokens=500,
            groq_api_key=GROQ_API_KEY
        )
        self.judge_chain = self._build_judge_chain()
        print("✅ LLM-as-Judge initialized!")

    def _build_judge_chain(self):
        """Build the judge chain."""

        prompt_template = """
        You are an expert evaluator for a movie content safety classifier.
        
        Your task is to evaluate the quality of a classification response.
        
        **Movie:**
        - Title: {title}
        - Overview: {overview}
        - Genres: {genres}
        - Rating: {rating}
        
        **Expected Classification:** {expected}
        **Reason:** {expected_reason}
        
        **System's Response:**
        {actual}
        
        **Evaluate the response on these dimensions (1-5 scale):**
        
        1. **Correctness** (1-5):
           - Does the classification match the expected? (5=exact match, 1=completely wrong)
        
        2. **Reasoning** (1-5):
           - Is the explanation logical and well-supported? (5=excellent, 1=poor)
        
        3. **Safety** (1-5):
           - Does it err on the side of caution when uncertain? (5=very cautious, 1=reckless)
        
        4. **Clarity** (1-5):
           - Is the response clear and well-formatted? (5=crystal clear, 1=confusing)
        
        Return ONLY a JSON object with this exact structure:
        {{
            "correctness": <int 1-5>,
            "reasoning": <int 1-5>,
            "safety": <int 1-5>,
            "clarity": <int 1-5>,
            "feedback": "<brief text feedback>"
        }}
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["title", "overview", "genres", "rating",
                             "expected", "expected_reason", "actual"]
        )

        chain = prompt | self.llm | JsonOutputParser()
        return chain

    def evaluate(
            self,
            title: str,
            overview: str,
            genres: str,
            rating: str,
            expected: str,
            expected_reason: str,
            actual: str
    ) -> JudgeScore:
        """Evaluate a single classification."""
        try:
            result = self.judge_chain.invoke({
                "title": title,
                "overview": overview,
                "genres": genres,
                "rating": rating,
                "expected": expected,
                "expected_reason": expected_reason,
                "actual": actual
            })

            overall = (
                              result["correctness"] +
                              result["reasoning"] +
                              result["safety"] +
                              result["clarity"]
                      ) / 4.0

            return JudgeScore(
                correctness=result["correctness"],
                reasoning=result["reasoning"],
                safety=result["safety"],
                clarity=result["clarity"],
                overall=overall,
                feedback=result.get("feedback", "")
            )
        except Exception as e:
            print(f"⚠️ Judge error: {e}")
            return JudgeScore(0, 0, 0, 0, 0.0, f"Error: {e}")


def main():
    """Test the LLM-as-Judge."""
    judge = LLMJudge()

    # Test case
    score = judge.evaluate(
        title="The Lion King",
        overview="A young lion prince flees his kingdom after the murder of his father.",
        genres="Animation, Adventure, Drama",
        rating="8.5",
        expected="Safe",
        expected_reason="Animated, family-friendly, positive messages",
        actual="Classification: Safe for children. Explanation: The Lion King is an animated film with positive themes about responsibility and friendship."
    )

    print(f"\n📊 Judge Score:")
    print(f"  Correctness: {score.correctness}/5")
    print(f"  Reasoning:   {score.reasoning}/5")
    print(f"  Safety:      {score.safety}/5")
    print(f"  Clarity:     {score.clarity}/5")
    print(f"  Overall:     {score.overall:.2f}/5")
    print(f"  Feedback:    {score.feedback}")


if __name__ == "__main__":
    main()