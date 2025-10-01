# -*- coding: utf-8 -*-

CHATBOT_GRADER_SYSTEM_PROMPT = """당신은 검색된 문서가 사용자의 질문과 얼마나 관련이 있는지 평가하는 평가자입니다.
문서에 질문과 관련된 키워드나 의미가 포함되어 있다면 '관련 있음'으로 평가하세요.
문서가 질문과 관련이 있는지 여부를 'yes' 또는 'no'의 이진 점수로 평가해주세요."""

CHATBOT_GRADE_PROMPT_TEMPLATE = """검색된 문서:

{document}

사용자 질문: {question}"""

CHATBOT_REWRITER_SYSTEM_PROMPT = """당신은 입력된 질문을 웹 검색에 최적화된 더 나은 버전으로 재작성하는 질문 재작성기입니다.
입력을 보고 근본적인 의미나 의도를 파악하여 질문을 다시 작성해주세요."""

CHATBOT_REWRITE_PROMPT_TEMPLATE = """다음은 초기 질문입니다:

{question}

더 나은 질문을 만들어주세요."""
