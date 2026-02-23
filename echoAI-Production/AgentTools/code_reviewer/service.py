"""
Code Review Service
Provides AI-powered code review using LLM for quality, security, and best practices analysis
"""
import logging
from typing import Dict, Any, Optional

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    SystemMessage = None
    HumanMessage = None

logger = logging.getLogger(__name__)


class CodeReviewService:
    """Service for AI-powered code review"""

    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            logger.warning("langchain_openai not installed. Code review will not work. Install with: pip install langchain-openai")
            self.llm = None
            return

        # Initialize LLM with fixed configuration
        # self.llm = ChatOpenAI(
        #     base_url="http://10.188.100.130:8002/v1",
        #     model="gpt-oss:20b",
        #     api_key="ollama"
        # )
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            api_key="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"
        )

    def review_code(self, code: str, language: str = "python", focus_areas: Optional[str] = None) -> str:
        """
        Perform comprehensive code review

        Args:
            code: Code snippet to review
            language: Programming language (python, javascript, java, etc.)
            focus_areas: Specific areas to focus on (optional)

        Returns:
            Detailed code review with categorized feedback
        """
        if not LANGCHAIN_AVAILABLE or self.llm is None:
            raise RuntimeError("langchain_openai is not installed. Please install: pip install langchain-openai")

        if not code or not code.strip():
            raise ValueError("Code to review cannot be empty")

        try:
            # Build focus areas string if provided
            focus_section = ""
            if focus_areas:
                focus_section = f"\n\nPay special attention to: {focus_areas}"

            task_description = f"""
Review the following {language.upper()} code thoroughly and provide a comprehensive analysis.

CODE TO REVIEW:
```{language}
{code}
```
{focus_section}

Provide your review in the following structured format:

## 🎯 OVERVIEW
- Provide a brief 2-3 sentence summary of what the code does
- Overall code quality rating (1-10)

## 🔴 CRITICAL ISSUES (Must Fix)
- List any critical bugs, security vulnerabilities, or crashes
- Explain the impact and how to reproduce
- Provide specific fixes with code examples

## 🟡 WARNINGS (Should Fix)
- Logic errors or edge cases not handled
- Performance bottlenecks or inefficiencies
- Missing error handling or validation
- Anti-patterns or code smells

## 🟢 SUGGESTIONS (Nice to Have)
- Code readability improvements
- Better naming conventions
- Refactoring opportunities
- Design pattern recommendations

## 🛡️ SECURITY REVIEW
- SQL injection, XSS, or other injection vulnerabilities
- Insecure data handling (passwords, tokens, PII)
- Authentication/authorization issues
- Dependency vulnerabilities

## ⚡ PERFORMANCE ANALYSIS
- Time complexity issues (O(n²) where O(n) possible)
- Memory leaks or excessive memory usage
- Database query optimization opportunities
- Caching opportunities

## 📚 BEST PRACTICES
- Adherence to language-specific conventions (PEP 8, ESLint, etc.)
- Code organization and structure
- Documentation and comments quality
- Test coverage recommendations

## ✅ WHAT'S DONE WELL
- Highlight positive aspects of the code
- Good practices worth keeping
- Effective patterns used

## 🔧 REFACTORED CODE (if significant improvements needed)
- Provide an improved version of the code
- Explain key changes made

## 📊 SUMMARY SCORE
- Bugs & Correctness: X/10
- Security: X/10
- Performance: X/10
- Readability: X/10
- Maintainability: X/10
- Overall: X/10

Be thorough, constructive, and educational. Provide specific examples and actionable advice.
"""

            system_prompt = SystemMessage(
                content="""You are a distinguished Senior Software Architect and Code Review Expert with over 20 years of experience across multiple domains:

- Software Engineering Best Practices & Design Patterns
- Security Engineering & Vulnerability Assessment (OWASP Top 10, CVE analysis)
- Performance Optimization & Scalability
- Code Quality & Technical Debt Management
- Multiple Programming Languages & Frameworks

Your code reviews are:
✓ Thorough and systematic - you catch subtle bugs others miss
✓ Educational - you explain the "why" behind each suggestion
✓ Constructive - you balance criticism with positive reinforcement
✓ Actionable - you provide specific fixes with code examples
✓ Prioritized - you clearly distinguish critical vs. nice-to-have improvements

You have reviewed thousands of production codebases and mentored hundreds of developers. Your feedback has prevented countless production incidents and improved code quality across organizations.

Analyze code with the mindset of: "What could go wrong in production? How can this be better?"
"""
            )
            human_prompt = HumanMessage(content=task_description)

            logger.info(f"Performing code review for {language} code ({len(code)} characters)")
            response = self.llm.invoke([system_prompt, human_prompt])

            logger.info("Code review completed successfully")
            return response.content

        except Exception as e:
            logger.error(f"Failed to perform code review: {str(e)}")
            raise

    def quick_review(self, code: str, language: str = "python") -> str:
        """
        Perform a quick, focused code review (faster, less detailed)

        Args:
            code: Code snippet to review
            language: Programming language

        Returns:
            Concise code review with key issues only
        """
        if not LANGCHAIN_AVAILABLE or self.llm is None:
            raise RuntimeError("langchain_openai is not installed. Please install: pip install langchain-openai")

        if not code or not code.strip():
            raise ValueError("Code to review cannot be empty")

        try:
            task_description = f"""
Perform a QUICK code review of this {language.upper()} code. Focus only on:
1. Critical bugs that will cause crashes or incorrect behavior
2. Security vulnerabilities (injection, auth issues, data leaks)
3. Major performance problems

CODE:
```{language}
{code}
```

Provide a concise review with:
- 🔴 Critical Issues (must fix)
- 🟡 Important Warnings (should fix)
- Overall assessment (1-2 sentences)

Be brief and direct. Only mention the most important issues.
"""

            system_prompt = SystemMessage(
                content="You are a senior code reviewer. Provide quick, focused feedback on critical issues only. Be concise and actionable."
            )
            human_prompt = HumanMessage(content=task_description)

            logger.info(f"Performing quick code review for {language} code")
            response = self.llm.invoke([system_prompt, human_prompt])

            return response.content

        except Exception as e:
            logger.error(f"Failed to perform quick code review: {str(e)}")
            raise
