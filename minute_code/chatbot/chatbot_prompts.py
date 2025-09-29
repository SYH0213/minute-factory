# -*- coding: utf-8 -*-

CHATBOT_GRADER_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

CHATBOT_GRADE_PROMPT_TEMPLATE = """Retrieved document: 

 {document} 

 User question: {question}"""

CHATBOT_REWRITER_SYSTEM_PROMPT = """You a question re-writer that converts an input question to a better version that is optimized
for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

CHATBOT_REWRITE_PROMPT_TEMPLATE = """Here is the initial question: 

 {question} 
 Formulate an improved question."""
