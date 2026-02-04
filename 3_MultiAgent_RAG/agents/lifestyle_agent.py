"""
Lifestyle Coach Agent for the Multi-Agent CKD System.

Provides guidance on:
- Exercise recommendations
- Hydration
- Blood pressure monitoring
- Smoking cessation
- Stress management
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CKD_STAGES

logger = logging.getLogger(__name__)


class LifestyleCategory(Enum):
    """Categories of lifestyle guidance."""
    EXERCISE = "exercise"
    HYDRATION = "hydration"
    BLOOD_PRESSURE = "blood_pressure"
    SMOKING = "smoking"
    STRESS = "stress"
    SLEEP = "sleep"
    WEIGHT = "weight"
    GENERAL = "general"


@dataclass
class LifestyleRecommendation:
    """A lifestyle recommendation."""
    category: LifestyleCategory
    title: str
    guidance: str
    tips: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class LifestyleAgentResponse:
    """Response from the Lifestyle Agent."""
    recommendations: list[LifestyleRecommendation]
    summary: str
    ckd_stage: Optional[int]
    confidence: float
    agent_name: str = "Lifestyle Coach"
    disclaimer: str = (
        "These are general lifestyle recommendations. Individual needs vary. "
        "Consult your healthcare team before starting new exercise programs "
        "or making significant lifestyle changes."
    )


class LifestyleAgent:
    """
    Lifestyle guidance agent for CKD patients.

    Provides evidence-based recommendations for:
    - Physical activity and exercise
    - Fluid intake and hydration
    - Blood pressure monitoring
    - Smoking cessation
    - Stress management
    - Sleep hygiene
    - Weight management

    Example:
        >>> agent = LifestyleAgent()
        >>> response = agent.get_guidance("exercise", ckd_stage=3)
        >>> print(response.recommendations[0].guidance)
    """

    # Lifestyle recommendations database
    RECOMMENDATIONS = {
        LifestyleCategory.EXERCISE: {
            "title": "Physical Activity for CKD",
            "general": (
                "Regular physical activity is beneficial and safe for most CKD patients. "
                "Exercise can help control blood pressure, maintain healthy weight, "
                "improve cardiovascular health, and enhance quality of life."
            ),
            "tips": [
                "Aim for 150 minutes of moderate activity per week",
                "Start slowly and gradually increase intensity",
                "Walking, swimming, and cycling are excellent choices",
                "Include strength training 2-3 times per week",
                "Listen to your body and rest when needed",
                "Exercise at the same time each day to build habit",
            ],
            "stage_specific": {
                1: "Most activities are safe. Focus on building consistent exercise habits.",
                2: "Continue regular activity. Monitor how you feel during exercise.",
                3: "Moderate exercise recommended. Avoid extreme exertion. Stay hydrated.",
                4: "Gentle exercise beneficial. Walking, light swimming. Avoid strenuous activity.",
                5: "Light activity as tolerated. Chair exercises, gentle walking. Consult team.",
            },
            "warnings": [
                "Stop if you experience chest pain, severe shortness of breath, or dizziness",
                "Avoid exercise if you have uncontrolled high blood pressure",
                "If on dialysis, coordinate exercise timing with treatments",
            ],
        },
        LifestyleCategory.HYDRATION: {
            "title": "Fluid Intake and Hydration",
            "general": (
                "Fluid needs vary significantly based on CKD stage and whether you're on dialysis. "
                "In early CKD, staying well-hydrated is important. In advanced CKD, "
                "fluid restriction may be necessary."
            ),
            "tips": [
                "Monitor your urine color - pale yellow indicates good hydration",
                "Weigh yourself daily to track fluid balance",
                "Spread fluid intake throughout the day",
                "Count all liquids including soups, ice cream, and juicy fruits",
                "Use smaller cups to help control intake if restricted",
            ],
            "stage_specific": {
                1: "Drink 6-8 glasses of water daily. Stay well hydrated.",
                2: "Maintain good hydration. 6-8 glasses daily unless advised otherwise.",
                3: "Follow your doctor's fluid recommendations. Usually 6-8 glasses.",
                4: "May need fluid restriction. Follow your healthcare team's guidance.",
                5: "Strict fluid restriction often required. Track all fluid intake carefully.",
            },
            "warnings": [
                "On dialysis: strict fluid limits are critical between treatments",
                "Watch for signs of fluid overload: swelling, shortness of breath",
                "Excessive fluid restriction can also be harmful - follow medical advice",
            ],
        },
        LifestyleCategory.BLOOD_PRESSURE: {
            "title": "Blood Pressure Management",
            "general": (
                "Blood pressure control is crucial in CKD. High blood pressure both causes "
                "and accelerates kidney damage. Target is usually below 130/80 mmHg."
            ),
            "tips": [
                "Check blood pressure at the same time daily",
                "Rest for 5 minutes before measuring",
                "Use a validated home blood pressure monitor",
                "Keep a log to share with your healthcare team",
                "Take medications at consistent times",
                "Reduce sodium intake to help lower blood pressure",
            ],
            "stage_specific": {
                1: "Target <130/80. Lifestyle changes may be sufficient.",
                2: "Target <130/80. May need medication alongside lifestyle changes.",
                3: "Target <130/80. Medication usually required. Monitor closely.",
                4: "Strict BP control essential. Multiple medications often needed.",
                5: "BP targets may vary. Follow nephrologist recommendations closely.",
            },
            "warnings": [
                "Don't stop blood pressure medications without consulting your doctor",
                "Report persistent readings above 180/120 immediately",
                "Dizziness when standing may indicate BP is too low",
            ],
        },
        LifestyleCategory.SMOKING: {
            "title": "Smoking Cessation",
            "general": (
                "Smoking accelerates CKD progression and dramatically increases cardiovascular risk. "
                "Quitting smoking is one of the most important things you can do for your kidney health."
            ),
            "tips": [
                "Set a quit date and tell friends/family for support",
                "Consider nicotine replacement therapy (patches, gum)",
                "Ask your doctor about prescription medications to help quit",
                "Identify triggers and plan alternatives",
                "Join a support group or quitline",
                "Remember: it's never too late to benefit from quitting",
            ],
            "stage_specific": {
                1: "Quitting now can slow CKD progression significantly.",
                2: "Quitting reduces cardiovascular risk and slows kidney decline.",
                3: "Critical to quit. Smoking accelerates progression to kidney failure.",
                4: "Quitting improves outcomes even at this stage.",
                5: "Improves dialysis outcomes and transplant eligibility.",
            },
            "warnings": [
                "Vaping/e-cigarettes are not safe alternatives",
                "Some quit-smoking medications need dose adjustment in CKD",
            ],
        },
        LifestyleCategory.STRESS: {
            "title": "Stress Management",
            "general": (
                "Living with CKD can be stressful. Chronic stress can raise blood pressure "
                "and affect overall health. Learning stress management techniques is valuable."
            ),
            "tips": [
                "Practice deep breathing exercises daily",
                "Try meditation or mindfulness apps",
                "Maintain social connections and support networks",
                "Engage in enjoyable hobbies and activities",
                "Consider counseling or support groups",
                "Establish regular sleep routines",
                "Limit news and social media if anxiety-provoking",
            ],
            "stage_specific": {
                1: "Build stress management habits early in your CKD journey.",
                2: "Regular stress relief helps maintain overall health.",
                3: "Stress management increasingly important as CKD advances.",
                4: "Preparing for potential dialysis can be stressful. Seek support.",
                5: "Support groups for dialysis patients can be very helpful.",
            },
            "warnings": [
                "Persistent anxiety or depression should be discussed with your doctor",
                "Some herbal stress remedies may be harmful to kidneys",
            ],
        },
        LifestyleCategory.SLEEP: {
            "title": "Sleep Health",
            "general": (
                "Quality sleep is important for overall health and blood pressure control. "
                "CKD patients often experience sleep problems including sleep apnea."
            ),
            "tips": [
                "Aim for 7-9 hours of sleep per night",
                "Maintain consistent sleep and wake times",
                "Create a cool, dark, quiet sleep environment",
                "Limit caffeine, especially in the afternoon",
                "Avoid screens for 1 hour before bed",
                "Report persistent sleep problems to your doctor",
            ],
            "stage_specific": {
                1: "Establish good sleep habits to support overall health.",
                2: "Quality sleep helps blood pressure control.",
                3: "Sleep disturbances may increase. Report to your doctor.",
                4: "Sleep apnea is common. Screening may be recommended.",
                5: "Sleep scheduling around dialysis can be challenging. Plan ahead.",
            },
            "warnings": [
                "Loud snoring and daytime sleepiness may indicate sleep apnea",
                "Some sleep aids are not safe in CKD - consult before using",
            ],
        },
        LifestyleCategory.WEIGHT: {
            "title": "Weight Management",
            "general": (
                "Maintaining a healthy weight helps control blood pressure and blood sugar, "
                "both important for kidney health. However, unintentional weight loss "
                "should be reported to your doctor."
            ),
            "tips": [
                "Aim for gradual, sustainable weight loss if overweight",
                "Focus on whole foods, vegetables, and lean proteins",
                "Combine healthy eating with regular physical activity",
                "Avoid crash diets or extreme restrictions",
                "Track your weight weekly, same time and conditions",
            ],
            "stage_specific": {
                1: "Achieving healthy weight can slow CKD progression.",
                2: "Weight management remains important for kidney protection.",
                3: "Balance weight goals with dietary restrictions for CKD.",
                4: "Work closely with dietitian for appropriate nutrition.",
                5: "Maintaining adequate nutrition becomes priority over weight loss.",
            },
            "warnings": [
                "Rapid weight changes may indicate fluid problems",
                "Unintentional weight loss should be evaluated",
                "Very low protein diets should only be followed under medical supervision",
            ],
        },
    }

    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the Lifestyle Agent.

        Args:
            llm: Optional LLM for generating personalized guidance
        """
        self.llm = llm
        logger.info("LifestyleAgent initialized")

    def can_handle(self, query: str) -> bool:
        """
        Check if this agent can handle the query.

        Args:
            query: User question

        Returns:
            True if query is about lifestyle/exercise
        """
        lifestyle_keywords = [
            "exercise", "activity", "walk", "swim", "gym",
            "water", "drink", "fluid", "hydration",
            "blood pressure", "bp", "hypertension",
            "smoking", "smoke", "quit", "cigarette",
            "stress", "anxiety", "relax", "meditation",
            "sleep", "insomnia", "tired", "fatigue",
            "weight", "overweight", "lose weight",
            "lifestyle", "healthy", "wellness",
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in lifestyle_keywords)

    def _detect_category(self, query: str) -> LifestyleCategory:
        """Detect the lifestyle category from query."""
        query_lower = query.lower()

        category_keywords = {
            LifestyleCategory.EXERCISE: ["exercise", "activity", "walk", "swim", "gym", "workout", "physical"],
            LifestyleCategory.HYDRATION: ["water", "drink", "fluid", "hydrat", "thirst"],
            LifestyleCategory.BLOOD_PRESSURE: ["blood pressure", "bp", "hypertension", "pressure"],
            LifestyleCategory.SMOKING: ["smok", "cigarette", "tobacco", "quit", "vape"],
            LifestyleCategory.STRESS: ["stress", "anxiety", "relax", "meditation", "mental", "cope"],
            LifestyleCategory.SLEEP: ["sleep", "insomnia", "tired", "fatigue", "rest", "nap"],
            LifestyleCategory.WEIGHT: ["weight", "overweight", "obese", "bmi", "lose weight"],
        }

        for category, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return category

        return LifestyleCategory.GENERAL

    def get_guidance(
        self,
        category: LifestyleCategory,
        ckd_stage: Optional[int] = None,
    ) -> LifestyleRecommendation:
        """
        Get guidance for a specific lifestyle category.

        Args:
            category: Lifestyle category
            ckd_stage: Patient's CKD stage

        Returns:
            LifestyleRecommendation
        """
        data = self.RECOMMENDATIONS.get(category)

        if data is None:
            return LifestyleRecommendation(
                category=category,
                title="General Lifestyle Guidance",
                guidance="Consult your healthcare team for specific lifestyle recommendations.",
                tips=[],
            )

        # Get stage-specific guidance
        guidance = data["general"]
        if ckd_stage and "stage_specific" in data:
            stage_guidance = data["stage_specific"].get(ckd_stage)
            if stage_guidance:
                guidance += f"\n\n**For CKD Stage {ckd_stage}:** {stage_guidance}"

        return LifestyleRecommendation(
            category=category,
            title=data["title"],
            guidance=guidance,
            tips=data.get("tips", []),
            warnings=data.get("warnings", []),
        )

    def answer(
        self,
        query: str,
        ckd_stage: Optional[int] = None,
    ) -> LifestyleAgentResponse:
        """
        Answer a lifestyle-related question.

        Args:
            query: User question about lifestyle
            ckd_stage: Patient's CKD stage

        Returns:
            LifestyleAgentResponse with recommendations
        """
        category = self._detect_category(query)
        recommendations = []

        if category == LifestyleCategory.GENERAL:
            # Provide overview of all categories
            for cat in [LifestyleCategory.EXERCISE, LifestyleCategory.BLOOD_PRESSURE,
                       LifestyleCategory.HYDRATION, LifestyleCategory.SMOKING]:
                rec = self.get_guidance(cat, ckd_stage)
                recommendations.append(rec)
            summary = self._generate_general_summary(ckd_stage)
        else:
            rec = self.get_guidance(category, ckd_stage)
            recommendations.append(rec)
            summary = self._generate_category_summary(category, ckd_stage)

        return LifestyleAgentResponse(
            recommendations=recommendations,
            summary=summary,
            ckd_stage=ckd_stage,
            confidence=0.85,
        )

    def _generate_category_summary(
        self,
        category: LifestyleCategory,
        ckd_stage: Optional[int],
    ) -> str:
        """Generate summary for a specific category."""
        data = self.RECOMMENDATIONS.get(category, {})
        title = data.get("title", category.value.replace("_", " ").title())

        summary = f"**{title}**\n\n"

        if ckd_stage:
            stage_desc = CKD_STAGES.get(ckd_stage, {}).get("description", "")
            summary += f"Recommendations for CKD Stage {ckd_stage} ({stage_desc}):\n\n"

        summary += data.get("general", "")[:200] + "..."

        return summary

    def _generate_general_summary(self, ckd_stage: Optional[int]) -> str:
        """Generate general lifestyle summary."""
        summary_parts = ["**Lifestyle Recommendations for CKD**\n"]

        if ckd_stage:
            stage_desc = CKD_STAGES.get(ckd_stage, {}).get("description", "")
            summary_parts.append(f"For Stage {ckd_stage}: {stage_desc}\n")

        summary_parts.extend([
            "\nKey areas for CKD management:",
            "- Regular physical activity",
            "- Blood pressure monitoring and control",
            "- Appropriate fluid intake",
            "- Smoking cessation",
            "- Stress management",
            "- Quality sleep",
            "\nWork with your healthcare team to personalize these recommendations.",
        ])

        return "\n".join(summary_parts)


def create_lifestyle_agent(llm: Optional[Any] = None) -> LifestyleAgent:
    """Factory function to create a Lifestyle agent."""
    return LifestyleAgent(llm=llm)
