"""
PII Detection and Anonymization for the CKD RAG System.

Uses Microsoft Presidio to detect and redact personally
identifiable information from user queries and responses.
"""

import logging
import re
from typing import Optional
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PII_ENTITIES

logger = logging.getLogger(__name__)


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""
    original_text: str
    anonymized_text: str
    pii_found: bool
    entities_detected: list[dict] = field(default_factory=list)
    placeholder_map: dict[str, str] = field(default_factory=dict)


class NHSNumberRecognizer(PatternRecognizer):
    """
    Custom recognizer for UK NHS numbers.

    NHS numbers are 10 digits, often formatted as:
    - 123 456 7890
    - 123-456-7890
    - 1234567890
    """

    PATTERNS = [
        Pattern(
            "NHS_SPACED",
            r"\b\d{3}\s\d{3}\s\d{4}\b",
            0.85,
        ),
        Pattern(
            "NHS_DASHED",
            r"\b\d{3}-\d{3}-\d{4}\b",
            0.85,
        ),
        Pattern(
            "NHS_PLAIN",
            r"\b\d{10}\b",
            0.6,  # Lower confidence as could be other numbers
        ),
    ]

    def __init__(self):
        super().__init__(
            supported_entity="UK_NHS",
            patterns=self.PATTERNS,
            context=["nhs", "nhs number", "health service", "patient"],
        )


class MedicalIDRecognizer(PatternRecognizer):
    """
    Custom recognizer for medical/patient IDs.

    Matches common patterns like:
    - MRN: 123456
    - Patient ID: ABC123456
    - Hospital No: H-12345
    """

    PATTERNS = [
        Pattern(
            "MRN_LABELED",
            r"(?:MRN|mrn|medical record)[\s:]*([A-Z0-9]{5,12})",
            0.9,
        ),
        Pattern(
            "PATIENT_ID_LABELED",
            r"(?:patient\s*(?:id|number|no))[\s:]*([A-Z0-9]{5,12})",
            0.9,
        ),
        Pattern(
            "HOSPITAL_NO",
            r"(?:hospital\s*(?:no|number))[\s:]*([A-Z]?-?\d{4,8})",
            0.85,
        ),
    ]

    def __init__(self):
        super().__init__(
            supported_entity="MEDICAL_LICENSE",
            patterns=self.PATTERNS,
            context=["patient", "medical", "hospital", "record"],
        )


class PIIHandler:
    """
    Handler for PII detection and anonymization.

    Supports:
    - Standard PII (names, emails, phones, dates)
    - UK-specific entities (NHS numbers)
    - Medical IDs (MRN, patient IDs)

    Example:
        >>> handler = PIIHandler()
        >>> result = handler.anonymize("My name is John Smith, NHS: 123-456-7890")
        >>> print(result.anonymized_text)
        "My name is <PERSON_1>, NHS: <UK_NHS_1>"
        >>> restored = handler.deanonymize(result)
        "My name is John Smith, NHS: 123-456-7890"
    """

    # Mapping of entity types to placeholder format
    PLACEHOLDER_FORMAT = {
        "PERSON": "<PERSON_{}>",
        "EMAIL_ADDRESS": "<EMAIL_{}>",
        "PHONE_NUMBER": "<PHONE_{}>",
        "UK_NHS": "<NHS_{}>",
        "DATE_TIME": "<DATE_{}>",
        "LOCATION": "<LOCATION_{}>",
        "MEDICAL_LICENSE": "<MEDICAL_ID_{}>",
        "CREDIT_CARD": "<CREDIT_CARD_{}>",
        "IP_ADDRESS": "<IP_{}>",
    }

    def __init__(
        self,
        entities: Optional[list[str]] = None,
        language: str = "en",
        score_threshold: float = 0.5,
    ):
        """
        Initialize the PII handler.

        Args:
            entities: List of entity types to detect (default from config)
            language: Language for NLP processing
            score_threshold: Minimum confidence score for detection
        """
        self.entities = entities or PII_ENTITIES
        self.language = language
        self.score_threshold = score_threshold

        # Initialize NLP engine
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": "en_core_web_sm"}],
        })

        nlp_engine = provider.create_engine()

        # Initialize analyzer with custom recognizers
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

        # Add custom recognizers
        self.analyzer.registry.add_recognizer(NHSNumberRecognizer())
        self.analyzer.registry.add_recognizer(MedicalIDRecognizer())

        # Initialize anonymizer
        self.anonymizer = AnonymizerEngine()

        logger.info(f"PIIHandler initialized with entities: {self.entities}")

    def detect(self, text: str) -> list[dict]:
        """
        Detect PII entities in text.

        Args:
            text: Text to analyze

        Returns:
            List of detected entities with type, value, and score
        """
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language=self.language,
            score_threshold=self.score_threshold,
        )

        entities = []
        for result in results:
            entities.append({
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": result.score,
                "value": text[result.start:result.end],
            })

        return entities

    def anonymize(
        self,
        text: str,
        use_placeholders: bool = True,
    ) -> PIIDetectionResult:
        """
        Detect and anonymize PII in text.

        Args:
            text: Text to anonymize
            use_placeholders: Use numbered placeholders (vs generic <PII>)

        Returns:
            PIIDetectionResult with anonymized text and mapping
        """
        # Detect entities
        entities = self.detect(text)

        if not entities:
            return PIIDetectionResult(
                original_text=text,
                anonymized_text=text,
                pii_found=False,
                entities_detected=[],
                placeholder_map={},
            )

        # Sort entities by position (reverse order for replacement)
        sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

        anonymized_text = text
        placeholder_map = {}
        entity_counters: dict[str, int] = {}

        for entity in sorted_entities:
            entity_type = entity["entity_type"]
            original_value = entity["value"]

            # Generate placeholder
            if use_placeholders:
                count = entity_counters.get(entity_type, 0) + 1
                entity_counters[entity_type] = count

                placeholder_template = self.PLACEHOLDER_FORMAT.get(
                    entity_type, "<{}_{{}}>"
                )
                if "{}" not in placeholder_template:
                    placeholder_template = f"<{entity_type}_{{}}>".format
                    placeholder = f"<{entity_type}_{count}>"
                else:
                    placeholder = placeholder_template.format(count)
            else:
                placeholder = f"<{entity_type}>"

            # Store mapping for potential de-anonymization
            placeholder_map[placeholder] = original_value

            # Replace in text
            anonymized_text = (
                anonymized_text[:entity["start"]]
                + placeholder
                + anonymized_text[entity["end"]:]
            )

        return PIIDetectionResult(
            original_text=text,
            anonymized_text=anonymized_text,
            pii_found=True,
            entities_detected=entities,
            placeholder_map=placeholder_map,
        )

    def deanonymize(self, result: PIIDetectionResult) -> str:
        """
        Restore original values from anonymized text.

        Args:
            result: PIIDetectionResult from anonymize()

        Returns:
            Text with original PII restored
        """
        text = result.anonymized_text

        for placeholder, original in result.placeholder_map.items():
            text = text.replace(placeholder, original)

        return text

    def anonymize_for_llm(self, text: str) -> tuple[str, dict]:
        """
        Prepare text for LLM processing by anonymizing PII.

        Returns both the anonymized text and the mapping needed
        to restore original values in the response.

        Args:
            text: User input text

        Returns:
            Tuple of (anonymized_text, placeholder_map)
        """
        result = self.anonymize(text)
        return result.anonymized_text, result.placeholder_map

    def restore_in_response(
        self,
        response: str,
        placeholder_map: dict[str, str],
    ) -> str:
        """
        Restore original PII values in LLM response.

        If the LLM's response contains placeholders, this restores
        the original values.

        Args:
            response: LLM response potentially containing placeholders
            placeholder_map: Mapping from anonymize_for_llm()

        Returns:
            Response with original values restored
        """
        for placeholder, original in placeholder_map.items():
            response = response.replace(placeholder, original)

        return response

    def get_summary(self, result: PIIDetectionResult) -> str:
        """
        Get a human-readable summary of detected PII.

        Args:
            result: PIIDetectionResult

        Returns:
            Summary string
        """
        if not result.pii_found:
            return "No PII detected."

        entity_counts: dict[str, int] = {}
        for entity in result.entities_detected:
            entity_type = entity["entity_type"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        parts = [f"{count} {etype}" for etype, count in entity_counts.items()]
        return f"PII detected: {', '.join(parts)}"


def create_pii_handler(
    score_threshold: float = 0.5,
) -> PIIHandler:
    """
    Factory function to create a PII handler.

    Args:
        score_threshold: Minimum confidence for detection

    Returns:
        Configured PIIHandler instance
    """
    return PIIHandler(score_threshold=score_threshold)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("Testing PIIHandler...")

    handler = PIIHandler()

    # Test cases
    test_texts = [
        "My name is John Smith and my email is john@example.com",
        "Please check my NHS number: 123-456-7890",
        "Patient ID: ABC123456, DOB: 15/03/1985",
        "What is the recommended potassium intake?",  # No PII
    ]

    for text in test_texts:
        print(f"\nOriginal: {text}")
        result = handler.anonymize(text)
        print(f"Anonymized: {result.anonymized_text}")
        print(f"Summary: {handler.get_summary(result)}")

        if result.pii_found:
            restored = handler.deanonymize(result)
            print(f"Restored: {restored}")
            assert restored == text, "De-anonymization failed!"
