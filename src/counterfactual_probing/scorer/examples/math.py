"""
Math answer correctness scorer - checks if generated answer matches ground truth.
"""

import re

from ..base import Scorer


class MathScorer(Scorer):
    """
    Scores math completions by checking if the final answer matches ground truth.

    Extracts the answer from \\boxed{} in the generated text and compares
    against the ground truth answer provided via set_context().

    Answer comparison includes normalization to handle:
    - Whitespace differences
    - Common LaTeX equivalences (\\frac vs plain fractions)
    - Numeric equivalence for simple numbers
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.ground_truth: str | None = None
        self.answer_field = (config or {}).get("answer_field", "answer")

    def set_context(self, prompt: str, metadata: dict) -> None:
        """
        Set the ground truth answer from metadata.

        Args:
            prompt: The original prompt text (unused)
            metadata: Must contain the answer field (default: 'answer')
        """
        self.ground_truth = metadata.get(self.answer_field)
        if self.ground_truth is None:
            # Try to extract from prompt if not in metadata
            self.ground_truth = ""

    def score(self, text: str) -> float:
        """
        Score a completion by checking answer correctness.

        Args:
            text: The generated completion text

        Returns:
            1.0 if answer matches ground truth, 0.0 otherwise
        """
        if self.ground_truth is None:
            return 0.0

        # Extract answer from generated text
        generated_answer = self._extract_boxed_answer(text)

        if not generated_answer:
            return 0.0

        # Compare answers
        if self._answers_match(generated_answer, self.ground_truth):
            return 1.0

        return 0.0

    def _extract_boxed_answer(self, text: str) -> str:
        """Extract the answer from \\boxed{...} in the text."""
        # Handle nested braces by finding all boxed patterns
        # and returning the last one (final answer)
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, text)

        if matches:
            return matches[-1].strip()

        # Try simpler pattern for non-nested cases
        simple_pattern = r'\\boxed\{([^}]+)\}'
        simple_matches = re.findall(simple_pattern, text)

        if simple_matches:
            return simple_matches[-1].strip()

        return ""

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        # Remove whitespace
        normalized = answer.strip()
        normalized = re.sub(r'\s+', '', normalized)

        # Normalize fractions FIRST: \frac{a}{b} -> a/b
        normalized = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', normalized)

        # Remove common LaTeX formatting
        normalized = normalized.replace('\\,', '')
        normalized = normalized.replace('\\;', '')
        normalized = normalized.replace('\\!', '')
        normalized = normalized.replace('\\text{', '')
        normalized = normalized.replace('\\mathrm{', '')
        normalized = normalized.replace('\\mathbf{', '')
        normalized = normalized.replace('\\left', '')
        normalized = normalized.replace('\\right', '')

        # Remove remaining braces
        normalized = normalized.replace('{', '')
        normalized = normalized.replace('}', '')

        # Normalize common symbols
        normalized = normalized.replace('\\cdot', '*')
        normalized = normalized.replace('\\times', '*')
        normalized = normalized.replace('\\div', '/')

        return normalized.lower()

    def _answers_match(self, generated: str, ground_truth: str) -> bool:
        """Check if two answers are equivalent."""
        # Normalize both answers
        gen_norm = self._normalize_answer(generated)
        gt_norm = self._normalize_answer(ground_truth)

        # Direct string match
        if gen_norm == gt_norm:
            return True

        # Try numeric comparison (handles fractions and decimals)
        gen_val = self._to_numeric(gen_norm)
        gt_val = self._to_numeric(gt_norm)

        return gen_val is not None and gt_val is not None and abs(gen_val - gt_val) < 1e-9

    def _to_numeric(self, s: str) -> float | None:
        """Convert a string to numeric value, handling fractions."""
        s = s.replace(',', '')

        # Try direct float conversion
        try:
            return float(s)
        except (ValueError, TypeError):
            pass

        # Try fraction evaluation
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, TypeError, ZeroDivisionError):
                    pass

        return None
