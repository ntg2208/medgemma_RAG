"""
Custom evaluation metrics for the CKD RAG System.

Implements CKD-specific metrics beyond standard RAGAS:
- Response time tracking
- Citation accuracy
- CKD stage appropriateness
- Medical disclaimer presence
- Actionability score
"""

import logging
import re
import time
from typing import Optional, Callable
from dataclasses import dataclass, field
from functools import wraps

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CKD_STAGES, DIETARY_LIMITS

logger = logging.getLogger(__name__)


@dataclass
class CKDMetricScores:
    """Container for custom CKD metric scores."""
    response_time_ms: float = 0.0
    citation_score: float = 0.0
    ckd_stage_appropriateness: float = 0.0
    disclaimer_present: bool = False
    actionability_score: float = 0.0
    medical_accuracy_indicators: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "response_time_ms": self.response_time_ms,
            "citation_score": self.citation_score,
            "ckd_stage_appropriateness": self.ckd_stage_appropriateness,
            "disclaimer_present": self.disclaimer_present,
            "actionability_score": self.actionability_score,
            "medical_accuracy_indicators": self.medical_accuracy_indicators,
        }


class CKDMetrics:
    """
    Custom evaluation metrics for CKD-specific RAG responses.

    Provides domain-specific quality assessment beyond
    generic RAG metrics.

    Example:
        >>> metrics = CKDMetrics()
        >>> scores = metrics.evaluate(
        ...     query="Potassium limits for stage 3?",
        ...     response="Limit to 2000-3000mg daily [Source: NICE NG203]",
        ...     ckd_stage=3,
        ...     response_time_ms=1500,
        ... )
        >>> print(scores.citation_score)
    """

    # Patterns for citation detection
    CITATION_PATTERNS = [
        r'\[Source:\s*[^\]]+\]',
        r'\[(?:NICE|KidneyCareUK)[^\]]*\]',
        r'\(Source:\s*[^)]+\)',
        r'according to (?:NICE|KidneyCareUK)',
        r'(?:NICE|KidneyCareUK) (?:guidelines?|recommends?)',
    ]

    # Medical disclaimer indicators
    DISCLAIMER_PATTERNS = [
        r'consult.*(?:doctor|healthcare|physician|provider)',
        r'medical advice',
        r'not (?:a )?substitute',
        r'professional guidance',
        r'individual.*may vary',
        r'speak (?:to|with).*(?:doctor|healthcare)',
    ]

    # Actionable language patterns
    ACTIONABLE_PATTERNS = [
        r'(?:you )?should',
        r'(?:it is )?recommended',
        r'try to',
        r'aim for',
        r'limit.*to',
        r'avoid',
        r'include',
        r'consider',
    ]

    # Stage-specific keywords that should appear
    STAGE_KEYWORDS = {
        1: ["monitor", "lifestyle", "blood pressure", "diabetes control"],
        2: ["monitor", "ace inhibitor", "arb", "blood pressure"],
        3: ["dietary", "potassium", "phosphorus", "medication review", "referral"],
        4: ["specialist", "dialysis preparation", "strict", "nephrologist"],
        5: ["dialysis", "transplant", "renal replacement", "end-stage"],
    }

    def __init__(self):
        """Initialize CKD metrics evaluator."""
        logger.info("CKDMetrics evaluator initialized")

    def measure_response_time(self, func: Callable) -> Callable:
        """
        Decorator to measure response time.

        Args:
            func: Function to measure

        Returns:
            Wrapped function that records timing
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Attach timing to result if possible
            if hasattr(result, 'response_time_ms'):
                result.response_time_ms = elapsed_ms

            return result, elapsed_ms

        return wrapper

    def evaluate_citations(self, response: str) -> float:
        """
        Evaluate citation quality in response.

        Args:
            response: Generated response text

        Returns:
            Citation score (0-1)
        """
        # Count citation matches
        citation_count = 0
        for pattern in self.CITATION_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            citation_count += len(matches)

        # Score based on presence and quantity
        if citation_count == 0:
            return 0.0
        elif citation_count == 1:
            return 0.6
        elif citation_count == 2:
            return 0.8
        else:
            return 1.0

    def evaluate_disclaimer(self, response: str) -> bool:
        """
        Check if response contains appropriate medical disclaimers.

        Args:
            response: Generated response text

        Returns:
            True if disclaimer present
        """
        response_lower = response.lower()

        for pattern in self.DISCLAIMER_PATTERNS:
            if re.search(pattern, response_lower):
                return True

        return False

    def evaluate_ckd_stage_appropriateness(
        self,
        response: str,
        ckd_stage: int,
    ) -> float:
        """
        Evaluate if response is appropriate for the CKD stage.

        Args:
            response: Generated response text
            ckd_stage: Patient's CKD stage (1-5)

        Returns:
            Appropriateness score (0-1)
        """
        if ckd_stage not in range(1, 6):
            return 0.5  # Neutral if stage unknown

        response_lower = response.lower()

        # Check for stage-appropriate keywords
        expected_keywords = self.STAGE_KEYWORDS.get(ckd_stage, [])
        found_keywords = sum(
            1 for kw in expected_keywords
            if kw in response_lower
        )

        keyword_score = found_keywords / len(expected_keywords) if expected_keywords else 0.5

        # Check dietary values if mentioned
        dietary_score = self._check_dietary_values(response, ckd_stage)

        # Combined score
        return (keyword_score + dietary_score) / 2

    def _check_dietary_values(self, response: str, ckd_stage: int) -> float:
        """Check if dietary values mentioned are appropriate for stage."""
        score = 1.0  # Start with perfect, deduct for errors

        # Check potassium values
        potassium_match = re.search(r'potassium.*?(\d{3,4})\s*(?:mg|milligrams?)', response, re.IGNORECASE)
        if potassium_match:
            value = int(potassium_match.group(1))
            limits = DIETARY_LIMITS["potassium"].get(ckd_stage, {})
            if limits:
                min_val = limits.get("min", 0)
                max_val = limits.get("max", 5000)
                if not (min_val <= value <= max_val + 500):  # Allow some tolerance
                    score -= 0.3

        # Check sodium values
        sodium_match = re.search(r'sodium.*?(\d{3,4})\s*(?:mg|milligrams?)', response, re.IGNORECASE)
        if sodium_match:
            value = int(sodium_match.group(1))
            limits = DIETARY_LIMITS["sodium"].get(ckd_stage, {})
            if limits:
                max_val = limits.get("max", 2300)
                if value > max_val + 300:  # Allow some tolerance
                    score -= 0.3

        return max(0.0, score)

    def evaluate_actionability(self, response: str) -> float:
        """
        Evaluate how actionable the response is.

        Args:
            response: Generated response text

        Returns:
            Actionability score (0-1)
        """
        response_lower = response.lower()

        actionable_count = 0
        for pattern in self.ACTIONABLE_PATTERNS:
            if re.search(pattern, response_lower):
                actionable_count += 1

        # Score based on actionable language
        if actionable_count == 0:
            return 0.2  # Some base score for informational responses
        elif actionable_count <= 2:
            return 0.6
        elif actionable_count <= 4:
            return 0.8
        else:
            return 1.0

    def check_medical_accuracy_indicators(
        self,
        response: str,
        query: str,
    ) -> dict[str, bool]:
        """
        Check for indicators of medical accuracy.

        This is a heuristic check, not a guarantee of accuracy.

        Args:
            response: Generated response text
            query: Original query

        Returns:
            Dict of accuracy indicators
        """
        indicators = {}
        response_lower = response.lower()
        query_lower = query.lower()

        # Check: Doesn't recommend NSAIDs (nephrotoxic)
        if "nsaid" in query_lower or "ibuprofen" in query_lower or "pain" in query_lower:
            indicators["avoids_nsaids"] = not bool(
                re.search(r'(?:take|use|recommend).*(?:nsaid|ibuprofen|naproxen)', response_lower)
            )

        # Check: Mentions eGFR appropriately
        if "egfr" in query_lower or "kidney function" in query_lower:
            indicators["mentions_egfr_correctly"] = "egfr" in response_lower

        # Check: Doesn't give specific doses without disclaimer
        has_specific_dose = bool(re.search(r'\d+\s*(?:mg|ml|mcg|units?)\s+(?:daily|twice|three times)', response_lower))
        has_consult_advice = self.evaluate_disclaimer(response)
        indicators["safe_dosing_advice"] = not has_specific_dose or has_consult_advice

        # Check: Recommends professional consultation for serious symptoms
        if any(word in query_lower for word in ["blood", "pain", "swelling", "emergency"]):
            indicators["advises_professional_consult"] = has_consult_advice

        return indicators

    def evaluate(
        self,
        query: str,
        response: str,
        ckd_stage: Optional[int] = None,
        response_time_ms: float = 0.0,
    ) -> CKDMetricScores:
        """
        Run all custom evaluations.

        Args:
            query: User question
            response: Generated response
            ckd_stage: Optional CKD stage
            response_time_ms: Response generation time

        Returns:
            CKDMetricScores with all evaluations
        """
        return CKDMetricScores(
            response_time_ms=response_time_ms,
            citation_score=self.evaluate_citations(response),
            ckd_stage_appropriateness=self.evaluate_ckd_stage_appropriateness(
                response, ckd_stage
            ) if ckd_stage else 0.5,
            disclaimer_present=self.evaluate_disclaimer(response),
            actionability_score=self.evaluate_actionability(response),
            medical_accuracy_indicators=self.check_medical_accuracy_indicators(
                response, query
            ),
        )


def create_ckd_metrics() -> CKDMetrics:
    """Factory function to create CKD metrics evaluator."""
    return CKDMetrics()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing CKDMetrics...")

    metrics = CKDMetrics()

    # Test response
    test_response = """
    For CKD stage 3, you should limit potassium intake to 2000-3000mg per day.
    High potassium foods to avoid include bananas, oranges, and potatoes.
    [Source: NICE NG203, Section 1.4]

    It's recommended to work with a dietitian for personalized advice.
    Please consult your healthcare provider before making significant dietary changes.
    """

    scores = metrics.evaluate(
        query="What are the potassium restrictions for CKD stage 3?",
        response=test_response,
        ckd_stage=3,
        response_time_ms=1500,
    )

    print("\nEvaluation Results:")
    for key, value in scores.to_dict().items():
        print(f"  {key}: {value}")
