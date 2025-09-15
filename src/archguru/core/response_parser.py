import re
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class ParsedResponse:
    """Structured parsed response"""
    recommendation: str
    reasoning: str
    trade_offs: List[str]
    implementation_steps: List[str]
    evidence: List[str]
    
class ResponseParser:
    """Centralized response parsing logic"""
    
    @staticmethod
    def parse_model_response(content: str) -> ParsedResponse:
        """Parse model response with fallback logic"""
        if not content or not content.strip():
            return ParsedResponse(
                recommendation="Error: Empty response",
                reasoning="Model returned empty content",
                trade_offs=[],
                implementation_steps=[],
                evidence=[]
            )
        
        content = content.strip()
        
        # Extract recommendation
        recommendation = ResponseParser._extract_recommendation(content)
        
        # Extract sections
        sections = ResponseParser._extract_sections(content)
        
        return ParsedResponse(
            recommendation=recommendation,
            reasoning=sections.get("reasoning", "Analysis based on research"),
            trade_offs=sections.get("trade_offs", []),
            implementation_steps=sections.get("implementation", []),
            evidence=sections.get("evidence", [])
        )
    
    @staticmethod
    def _extract_recommendation(content: str) -> str:
        """Extract recommendation with fallback"""
        # Try "Final Recommendation:" pattern
        match = re.search(r'Final Recommendation:\s*(.*?)(?:\n|$)', content, re.DOTALL)
        if match:
            return f"Final Recommendation: {match.group(1).strip()}"
        
        # Fallback to first meaningful line
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if lines:
            return lines[0][:200]
        return content[:200]
    
    @staticmethod
    def _extract_sections(content: str) -> dict:
        """Extract structured sections from content"""
        sections = {}
        current_section = None
        current_items = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            line_lower = line.lower()
            if line_lower.startswith('reasoning:'):
                if current_section and current_items:
                    sections[current_section] = current_items
                current_section = 'reasoning'
                current_items = []
            elif line_lower.startswith('trade-offs:'):
                if current_section and current_items:
                    sections[current_section] = current_items
                current_section = 'trade_offs'
                current_items = []
            elif line_lower.startswith('implementation'):
                if current_section and current_items:
                    sections[current_section] = current_items
                current_section = 'implementation'
                current_items = []
            elif line_lower.startswith('evidence:'):
                if current_section and current_items:
                    sections[current_section] = current_items
                current_section = 'evidence'
                current_items = []
            elif line.startswith('- ') and current_section:
                current_items.append(line[2:].strip())
            elif current_section == 'reasoning':
                current_items.append(line)
        
        # Save last section
        if current_section and current_items:
            sections[current_section] = current_items
            
        return sections

    @staticmethod
    def ensure_final_recommendation(text: str) -> str:
        """Ensure text starts with 'Final Recommendation:'"""
        if not text:
            return "Final Recommendation: No consensus reached."
        
        text = text.strip()
        if text.lower().startswith("final recommendation:"):
            return text
            
        # Extract first meaningful sentence
        for line in text.splitlines():
            line = line.strip()
            if line and not line.endswith(':'):
                sentence = line.split('.')[0].strip()
                if sentence:
                    return f"Final Recommendation: {sentence[:160]}."
        
        return "Final Recommendation: No consensus reached."