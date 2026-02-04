"""
Diet Calculator Agent for the Multi-Agent CKD System.

Provides personalized dietary recommendations based on
CKD stage, weight, and individual factors.
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DIETARY_LIMITS, CKD_STAGES

logger = logging.getLogger(__name__)


@dataclass
class DietaryRecommendation:
    """Single dietary recommendation."""
    nutrient: str
    daily_limit: str
    unit: str
    guidance: str
    foods_to_limit: list[str] = field(default_factory=list)
    foods_to_prefer: list[str] = field(default_factory=list)


@dataclass
class DietAgentResponse:
    """Response from the Diet Agent."""
    recommendations: list[DietaryRecommendation]
    summary: str
    ckd_stage: int
    weight_kg: Optional[float]
    confidence: float
    agent_name: str = "Diet Agent"
    disclaimer: str = (
        "These are general guidelines based on your CKD stage. "
        "Individual needs may vary. Please consult a registered dietitian "
        "for personalized meal planning."
    )


class DietAgent:
    """
    Dietary recommendation agent for CKD patients.

    Calculates personalized daily limits for:
    - Potassium
    - Phosphorus
    - Sodium
    - Protein

    Based on CKD stage and body weight.

    Example:
        >>> agent = DietAgent()
        >>> response = agent.calculate(ckd_stage=3, weight_kg=70)
        >>> for rec in response.recommendations:
        ...     print(f"{rec.nutrient}: {rec.daily_limit}")
    """

    # Foods database for recommendations
    FOODS_DATABASE = {
        "potassium": {
            "high": [
                "bananas", "oranges", "potatoes", "tomatoes", "spinach",
                "avocados", "dried fruits", "beans", "nuts", "chocolate",
            ],
            "low": [
                "apples", "berries", "grapes", "pineapple", "cabbage",
                "carrots", "cauliflower", "cucumber", "lettuce", "onions",
            ],
        },
        "phosphorus": {
            "high": [
                "dairy products", "processed meats", "cola drinks",
                "nuts", "seeds", "whole grains", "chocolate", "beer",
            ],
            "low": [
                "fresh fruits", "fresh vegetables", "rice milk",
                "white bread", "pasta", "rice", "corn cereals",
            ],
        },
        "sodium": {
            "high": [
                "table salt", "processed foods", "canned soups",
                "deli meats", "fast food", "soy sauce", "pickles",
                "cheese", "bread", "pizza",
            ],
            "low": [
                "fresh fruits", "fresh vegetables", "unsalted nuts",
                "fresh meat", "fresh fish", "herbs and spices",
            ],
        },
        "protein": {
            "high_quality": [
                "eggs", "fish", "poultry", "lean meat", "dairy",
            ],
            "plant_based": [
                "beans", "lentils", "tofu", "tempeh", "quinoa",
            ],
        },
    }

    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the Diet Agent.

        Args:
            llm: Optional LLM for generating explanations
        """
        self.llm = llm
        logger.info("DietAgent initialized")

    def can_handle(self, query: str) -> bool:
        """
        Check if this agent can handle the query.

        Args:
            query: User question

        Returns:
            True if query is about diet/nutrition
        """
        diet_keywords = [
            "diet", "food", "eat", "nutrition", "potassium", "phosphorus",
            "sodium", "salt", "protein", "meal", "limit", "avoid",
            "how much", "daily", "intake", "restrict",
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in diet_keywords)

    def _get_nutrient_limits(
        self,
        ckd_stage: int,
        weight_kg: Optional[float] = None,
    ) -> dict:
        """Get nutrient limits for a CKD stage."""
        limits = {}

        for nutrient, stage_limits in DIETARY_LIMITS.items():
            if ckd_stage in stage_limits:
                limits[nutrient] = stage_limits[ckd_stage].copy()

        # Adjust protein for body weight
        if weight_kg and "protein" in limits:
            min_g = limits["protein"]["min"] * weight_kg
            max_g = limits["protein"]["max"] * weight_kg
            limits["protein"]["calculated_min"] = round(min_g, 1)
            limits["protein"]["calculated_max"] = round(max_g, 1)

        return limits

    def _create_recommendation(
        self,
        nutrient: str,
        limits: dict,
        ckd_stage: int,
    ) -> DietaryRecommendation:
        """Create a dietary recommendation for a nutrient."""
        unit = limits.get("unit", "mg")

        # Format the daily limit string
        if "calculated_min" in limits:
            # Protein with weight calculation
            daily_limit = f"{limits['calculated_min']}-{limits['calculated_max']}g"
            unit = "g (based on body weight)"
        elif "min" in limits and "max" in limits:
            daily_limit = f"{limits['min']}-{limits['max']}"
        elif "max" in limits:
            daily_limit = f"<{limits['max']}"
        else:
            daily_limit = "Consult dietitian"

        # Get food recommendations
        foods_db = self.FOODS_DATABASE.get(nutrient, {})

        if nutrient == "protein":
            foods_to_limit = []  # Don't limit, just monitor
            foods_to_prefer = foods_db.get("high_quality", [])[:5]
            guidance = self._get_protein_guidance(ckd_stage)
        else:
            foods_to_limit = foods_db.get("high", [])[:5]
            foods_to_prefer = foods_db.get("low", [])[:5]
            guidance = self._get_nutrient_guidance(nutrient, ckd_stage)

        return DietaryRecommendation(
            nutrient=nutrient.capitalize(),
            daily_limit=daily_limit,
            unit=unit,
            guidance=guidance,
            foods_to_limit=foods_to_limit,
            foods_to_prefer=foods_to_prefer,
        )

    def _get_nutrient_guidance(self, nutrient: str, ckd_stage: int) -> str:
        """Get guidance text for a nutrient."""
        guidance = {
            "potassium": {
                1: "Monitor potassium levels. Most people can maintain normal intake.",
                2: "Monitor potassium levels. Most people can maintain normal intake.",
                3: "Begin limiting high-potassium foods. Check levels regularly.",
                4: "Strict potassium restriction recommended. Work with dietitian.",
                5: "Very strict potassium control needed. Monitor closely.",
            },
            "phosphorus": {
                1: "Maintain normal phosphorus intake from whole foods.",
                2: "Maintain normal phosphorus intake. Limit processed foods.",
                3: "Limit phosphorus. Avoid processed foods with phosphate additives.",
                4: "Strict phosphorus restriction. May need phosphate binders.",
                5: "Very strict phosphorus control. Use phosphate binders as prescribed.",
            },
            "sodium": {
                1: "Aim for less than 2,300mg daily. Reduce processed foods.",
                2: "Aim for less than 2,300mg daily. Reduce processed foods.",
                3: "Limit to 2,000mg daily. Avoid adding salt to food.",
                4: "Strict sodium restriction to help control blood pressure.",
                5: "Very strict sodium control, especially if on dialysis.",
            },
        }

        return guidance.get(nutrient, {}).get(
            ckd_stage,
            f"Consult your healthcare team for {nutrient} guidance."
        )

    def _get_protein_guidance(self, ckd_stage: int) -> str:
        """Get protein-specific guidance."""
        guidance = {
            1: "Maintain moderate protein intake from high-quality sources.",
            2: "Maintain moderate protein intake from high-quality sources.",
            3: "Consider moderate protein restriction (0.6-0.8g/kg). Focus on quality.",
            4: "Protein restriction important. Emphasize high-quality protein sources.",
            5: "On dialysis: protein needs increase. Pre-dialysis: restrict moderately.",
        }
        return guidance.get(ckd_stage, "Consult dietitian for protein guidance.")

    def calculate(
        self,
        ckd_stage: int,
        weight_kg: Optional[float] = None,
        specific_nutrient: Optional[str] = None,
    ) -> DietAgentResponse:
        """
        Calculate dietary recommendations.

        Args:
            ckd_stage: CKD stage (1-5)
            weight_kg: Body weight in kg (for protein calculation)
            specific_nutrient: Optional specific nutrient to focus on

        Returns:
            DietAgentResponse with recommendations
        """
        if not 1 <= ckd_stage <= 5:
            logger.warning(f"Invalid CKD stage: {ckd_stage}")
            ckd_stage = 3  # Default to stage 3

        limits = self._get_nutrient_limits(ckd_stage, weight_kg)
        recommendations = []

        nutrients_to_include = (
            [specific_nutrient] if specific_nutrient
            else ["potassium", "phosphorus", "sodium", "protein"]
        )

        for nutrient in nutrients_to_include:
            if nutrient in limits:
                rec = self._create_recommendation(nutrient, limits[nutrient], ckd_stage)
                recommendations.append(rec)

        # Generate summary
        stage_desc = CKD_STAGES.get(ckd_stage, {}).get("description", "")
        summary = self._generate_summary(ckd_stage, stage_desc, recommendations)

        return DietAgentResponse(
            recommendations=recommendations,
            summary=summary,
            ckd_stage=ckd_stage,
            weight_kg=weight_kg,
            confidence=0.85,
        )

    def _generate_summary(
        self,
        ckd_stage: int,
        stage_desc: str,
        recommendations: list[DietaryRecommendation],
    ) -> str:
        """Generate a summary of dietary recommendations."""
        summary_parts = [
            f"**Dietary Guidelines for CKD Stage {ckd_stage}**",
            f"({stage_desc})",
            "",
            "Key daily limits:",
        ]

        for rec in recommendations:
            summary_parts.append(f"- {rec.nutrient}: {rec.daily_limit} {rec.unit}")

        summary_parts.extend([
            "",
            "Focus on fresh, whole foods and limit processed items.",
            "Work with a registered dietitian for personalized guidance.",
        ])

        return "\n".join(summary_parts)

    def answer(
        self,
        query: str,
        ckd_stage: Optional[int] = None,
        weight_kg: Optional[float] = None,
    ) -> DietAgentResponse:
        """
        Answer a dietary question.

        Args:
            query: User question about diet
            ckd_stage: CKD stage (required for accurate advice)
            weight_kg: Body weight for protein calculation

        Returns:
            DietAgentResponse with recommendations
        """
        if ckd_stage is None:
            ckd_stage = 3  # Default assumption
            logger.info("No CKD stage provided, defaulting to stage 3")

        # Check for specific nutrient queries
        query_lower = query.lower()
        specific_nutrient = None

        for nutrient in ["potassium", "phosphorus", "sodium", "protein"]:
            if nutrient in query_lower:
                specific_nutrient = nutrient
                break

        return self.calculate(
            ckd_stage=ckd_stage,
            weight_kg=weight_kg,
            specific_nutrient=specific_nutrient,
        )


def create_diet_agent(llm: Optional[Any] = None) -> DietAgent:
    """Factory function to create a Diet agent."""
    return DietAgent(llm=llm)
