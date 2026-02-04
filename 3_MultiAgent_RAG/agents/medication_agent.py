"""
Medication Checker Agent for the Multi-Agent CKD System.

Provides kidney-safe medication guidance including:
- Nephrotoxic drug warnings
- Dose adjustment recommendations
- Drug interaction alerts
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level for medications in CKD."""
    SAFE = "safe"
    CAUTION = "caution"
    AVOID = "avoid"
    CONTRAINDICATED = "contraindicated"


@dataclass
class MedicationInfo:
    """Information about a medication for CKD patients."""
    name: str
    risk_level: RiskLevel
    ckd_considerations: str
    dose_adjustment: Optional[str] = None
    alternatives: list[str] = field(default_factory=list)
    interactions: list[str] = field(default_factory=list)


@dataclass
class MedicationAgentResponse:
    """Response from the Medication Agent."""
    medications_analyzed: list[MedicationInfo]
    general_guidance: str
    warnings: list[str]
    confidence: float
    agent_name: str = "Medication Agent"
    disclaimer: str = (
        "This information is for educational purposes only. "
        "ALWAYS consult your doctor or pharmacist before starting, stopping, "
        "or changing any medication. Never adjust doses without medical supervision."
    )


class MedicationAgent:
    """
    Medication safety agent for CKD patients.

    Provides guidance on:
    - Nephrotoxic medications to avoid
    - Dose adjustments for renal impairment
    - Safer alternatives
    - Drug interactions

    Example:
        >>> agent = MedicationAgent()
        >>> response = agent.check("ibuprofen", ckd_stage=3)
        >>> print(response.medications_analyzed[0].risk_level)
    """

    # Database of medications with CKD considerations
    MEDICATION_DATABASE = {
        # NSAIDs - Generally avoid in CKD
        "ibuprofen": MedicationInfo(
            name="Ibuprofen",
            risk_level=RiskLevel.AVOID,
            ckd_considerations="NSAIDs can reduce kidney blood flow and worsen kidney function. "
                              "Can cause acute kidney injury, especially with dehydration.",
            dose_adjustment="Avoid use in CKD stages 3-5. Use with extreme caution in stages 1-2.",
            alternatives=["paracetamol/acetaminophen", "topical treatments", "physical therapy"],
            interactions=["ACE inhibitors", "ARBs", "diuretics", "lithium"],
        ),
        "naproxen": MedicationInfo(
            name="Naproxen",
            risk_level=RiskLevel.AVOID,
            ckd_considerations="NSAID - can reduce kidney blood flow and worsen kidney function.",
            dose_adjustment="Avoid in CKD stages 3-5.",
            alternatives=["paracetamol/acetaminophen", "topical NSAIDs with caution"],
            interactions=["ACE inhibitors", "ARBs", "diuretics"],
        ),
        "aspirin": MedicationInfo(
            name="Aspirin",
            risk_level=RiskLevel.CAUTION,
            ckd_considerations="Low-dose aspirin for cardiovascular protection may be continued. "
                              "Higher doses for pain should be avoided.",
            dose_adjustment="Low-dose (75-100mg) generally safe. Avoid high doses.",
            alternatives=["paracetamol for pain"],
        ),

        # Pain medications
        "paracetamol": MedicationInfo(
            name="Paracetamol (Acetaminophen)",
            risk_level=RiskLevel.SAFE,
            ckd_considerations="Generally safe at recommended doses. First-line pain reliever for CKD.",
            dose_adjustment="Standard doses usually appropriate. May reduce maximum daily dose in severe CKD.",
            alternatives=[],
        ),
        "codeine": MedicationInfo(
            name="Codeine",
            risk_level=RiskLevel.CAUTION,
            ckd_considerations="Active metabolite (morphine) accumulates in kidney impairment. "
                              "Can cause prolonged sedation and respiratory depression.",
            dose_adjustment="Reduce dose by 50% in CKD stages 4-5. Increase dosing interval.",
            alternatives=["paracetamol", "tramadol with dose adjustment"],
        ),
        "tramadol": MedicationInfo(
            name="Tramadol",
            risk_level=RiskLevel.CAUTION,
            ckd_considerations="Accumulates in renal impairment. Increased risk of seizures.",
            dose_adjustment="Maximum 100mg twice daily in CKD stages 4-5. Extend dosing interval.",
            alternatives=["paracetamol"],
        ),

        # Antibiotics
        "metformin": MedicationInfo(
            name="Metformin",
            risk_level=RiskLevel.CAUTION,
            ckd_considerations="Risk of lactic acidosis increases with declining kidney function. "
                              "Now considered safe up to eGFR 30 with dose adjustment.",
            dose_adjustment="eGFR 30-45: max 1000mg/day. eGFR <30: contraindicated.",
            alternatives=["SGLT2 inhibitors (kidney protective)", "DPP-4 inhibitors"],
        ),
        "gentamicin": MedicationInfo(
            name="Gentamicin",
            risk_level=RiskLevel.AVOID,
            ckd_considerations="Highly nephrotoxic aminoglycoside antibiotic. "
                              "Can cause permanent kidney damage.",
            dose_adjustment="If essential: extend interval, monitor levels closely.",
            alternatives=["other antibiotic classes based on infection"],
        ),
        "nitrofurantoin": MedicationInfo(
            name="Nitrofurantoin",
            risk_level=RiskLevel.AVOID,
            ckd_considerations="Ineffective in CKD as inadequate urinary concentration. "
                              "Risk of peripheral neuropathy.",
            dose_adjustment="Avoid if eGFR <45.",
            alternatives=["trimethoprim", "fosfomycin", "other based on culture"],
        ),

        # Blood pressure medications
        "ace_inhibitor": MedicationInfo(
            name="ACE Inhibitors (e.g., ramipril, lisinopril)",
            risk_level=RiskLevel.SAFE,
            ckd_considerations="Kidney protective! Recommended for CKD with proteinuria. "
                              "May cause initial small rise in creatinine (up to 30% acceptable).",
            dose_adjustment="Start low, titrate up. Monitor potassium and creatinine.",
            interactions=["potassium supplements", "potassium-sparing diuretics", "NSAIDs"],
        ),
        "arb": MedicationInfo(
            name="ARBs (e.g., losartan, valsartan)",
            risk_level=RiskLevel.SAFE,
            ckd_considerations="Kidney protective! Alternative to ACE inhibitors. "
                              "Similar benefits for proteinuria reduction.",
            dose_adjustment="Start low, titrate up. Monitor potassium and creatinine.",
            interactions=["potassium supplements", "potassium-sparing diuretics", "NSAIDs"],
        ),

        # Other common medications
        "omeprazole": MedicationInfo(
            name="Omeprazole (PPI)",
            risk_level=RiskLevel.CAUTION,
            ckd_considerations="Long-term use associated with increased CKD risk. "
                              "May cause interstitial nephritis. Use lowest effective dose.",
            dose_adjustment="No specific adjustment, but minimize duration of use.",
            alternatives=["H2 blockers (famotidine)", "antacids for short-term use"],
        ),
        "gabapentin": MedicationInfo(
            name="Gabapentin",
            risk_level=RiskLevel.CAUTION,
            ckd_considerations="Renally excreted - accumulates in kidney impairment. "
                              "Can cause excessive sedation if not dose-adjusted.",
            dose_adjustment="eGFR 30-59: max 600mg/day. eGFR 15-29: max 300mg/day. eGFR <15: max 150mg/day.",
            alternatives=["pregabalin (also needs adjustment)", "amitriptyline"],
        ),
    }

    # Aliases for medication lookup
    MEDICATION_ALIASES = {
        "advil": "ibuprofen",
        "motrin": "ibuprofen",
        "nurofen": "ibuprofen",
        "aleve": "naproxen",
        "tylenol": "paracetamol",
        "acetaminophen": "paracetamol",
        "ramipril": "ace_inhibitor",
        "lisinopril": "ace_inhibitor",
        "enalapril": "ace_inhibitor",
        "losartan": "arb",
        "valsartan": "arb",
        "candesartan": "arb",
        "prilosec": "omeprazole",
        "nexium": "omeprazole",
        "neurontin": "gabapentin",
    }

    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the Medication Agent.

        Args:
            llm: Optional LLM for generating explanations
        """
        self.llm = llm
        logger.info("MedicationAgent initialized")

    def can_handle(self, query: str) -> bool:
        """
        Check if this agent can handle the query.

        Args:
            query: User question

        Returns:
            True if query is about medications
        """
        med_keywords = [
            "medication", "medicine", "drug", "pill", "tablet",
            "safe", "take", "avoid", "painkiller", "antibiotic",
            "prescription", "over-the-counter", "otc", "dose",
            "ibuprofen", "paracetamol", "aspirin", "nsaid",
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in med_keywords)

    def _lookup_medication(self, name: str) -> Optional[MedicationInfo]:
        """Look up medication in database."""
        name_lower = name.lower().strip()

        # Check aliases first
        if name_lower in self.MEDICATION_ALIASES:
            name_lower = self.MEDICATION_ALIASES[name_lower]

        # Look up in database
        if name_lower in self.MEDICATION_DATABASE:
            return self.MEDICATION_DATABASE[name_lower]

        # Partial match
        for key, info in self.MEDICATION_DATABASE.items():
            if name_lower in key or key in name_lower:
                return info

        return None

    def check(
        self,
        medication_name: str,
        ckd_stage: Optional[int] = None,
    ) -> MedicationAgentResponse:
        """
        Check a medication for CKD safety.

        Args:
            medication_name: Name of medication to check
            ckd_stage: Patient's CKD stage

        Returns:
            MedicationAgentResponse with safety information
        """
        med_info = self._lookup_medication(medication_name)
        warnings = []

        if med_info is None:
            # Unknown medication
            return MedicationAgentResponse(
                medications_analyzed=[],
                general_guidance=f"I don't have specific information about '{medication_name}' in my database. "
                                 "Please consult your pharmacist or doctor for guidance on this medication.",
                warnings=["Unknown medication - professional consultation required"],
                confidence=0.3,
            )

        # Add stage-specific warnings
        if ckd_stage:
            if med_info.risk_level == RiskLevel.AVOID and ckd_stage >= 3:
                warnings.append(f"⚠️ HIGH RISK: {med_info.name} should be avoided in CKD stage {ckd_stage}")
            elif med_info.risk_level == RiskLevel.CAUTION:
                warnings.append(f"⚠️ CAUTION: {med_info.name} requires careful monitoring in CKD")

        general_guidance = self._generate_guidance(med_info, ckd_stage)

        return MedicationAgentResponse(
            medications_analyzed=[med_info],
            general_guidance=general_guidance,
            warnings=warnings,
            confidence=0.85,
        )

    def _generate_guidance(
        self,
        med_info: MedicationInfo,
        ckd_stage: Optional[int],
    ) -> str:
        """Generate guidance text for a medication."""
        parts = [
            f"**{med_info.name}**",
            f"Risk Level: {med_info.risk_level.value.upper()}",
            "",
            med_info.ckd_considerations,
        ]

        if med_info.dose_adjustment:
            parts.extend(["", f"**Dose Adjustment:** {med_info.dose_adjustment}"])

        if med_info.alternatives:
            parts.extend(["", f"**Alternatives:** {', '.join(med_info.alternatives)}"])

        if med_info.interactions:
            parts.extend(["", f"**Watch for interactions with:** {', '.join(med_info.interactions)}"])

        return "\n".join(parts)

    def answer(
        self,
        query: str,
        ckd_stage: Optional[int] = None,
    ) -> MedicationAgentResponse:
        """
        Answer a medication-related question.

        Args:
            query: User question about medications
            ckd_stage: Patient's CKD stage

        Returns:
            MedicationAgentResponse with guidance
        """
        query_lower = query.lower()

        # Try to extract medication names from query
        for med_name in list(self.MEDICATION_DATABASE.keys()) + list(self.MEDICATION_ALIASES.keys()):
            if med_name in query_lower:
                return self.check(med_name, ckd_stage)

        # General medication question
        return MedicationAgentResponse(
            medications_analyzed=[],
            general_guidance=self._get_general_medication_guidance(ckd_stage),
            warnings=[],
            confidence=0.7,
        )

    def _get_general_medication_guidance(self, ckd_stage: Optional[int]) -> str:
        """Get general medication guidance for CKD patients."""
        guidance = """
**General Medication Safety for CKD Patients**

**Medications to AVOID or use with extreme caution:**
- NSAIDs (ibuprofen, naproxen) - can worsen kidney function
- Some antibiotics (gentamicin, nitrofurantoin)
- Certain diabetes medications at lower eGFR levels

**Generally SAFE medications:**
- Paracetamol/acetaminophen for pain (at recommended doses)
- ACE inhibitors and ARBs (actually kidney-protective!)

**Key principles:**
1. Always inform healthcare providers about your CKD
2. Check with pharmacist before any new medication (including OTC)
3. Avoid herbal supplements without medical advice
4. Stay hydrated, especially when taking medications
5. Report any changes in urine output or swelling
"""
        return guidance.strip()


def create_medication_agent(llm: Optional[Any] = None) -> MedicationAgent:
    """Factory function to create a Medication agent."""
    return MedicationAgent(llm=llm)
