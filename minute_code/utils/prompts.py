# -*- coding: utf-8 -*-

# --- GPT 텍스트 교정 프롬프트 ---
GPT_CORRECTION_SYSTEM_PROMPT = "You are a helpful assistant that corrects and refines meeting transcripts."
GPT_CORRECTION_USER_PROMPT = """다음 텍스트는 '{topic}'에 대한 회의 내용입니다. 
주요 키워드는 {keywords} 입니다. 
문맥에 맞게 문장을 다듬고, 맞춤법 및 띄어쓰기를 수정해주세요. 
특히, 키워드가 포함된 문장은 더 자연스럽게 만들어주세요.

원본 텍스트:
{text}

교정된 텍스트:
"""

# --- Gemini 텍스트 교정 프롬프트 ---
GEMINI_CORRECTION_PROMPT_TEMPLATE = '''
당신은 회의록을 교정하고 다듬는 전문적인 AI 어시스턴트입니다.
회의 주제: "{topic}"
주요 키워드: {keywords}

다음 원본 텍스트의 문맥을 유지하면서, 맞춤법, 띄어쓰기, 문법 오류를 수정해주세요.
특히, 주요 키워드가 포함된 문장은 더 자연스럽고 전문적인 표현으로 다듬어주세요.
결과는 교정된 텍스트만 남겨주세요.

원본 텍스트:
{text}

교정된 텍스트:
'''

# --- GPT 텍스트 요약 프롬프트 (청크별) ---
GPT_CHUNK_SUMMARY_SYSTEM_PROMPT = "You are a helpful assistant that summarizes parts of a meeting transcript."
GPT_CHUNK_SUMMARY_USER_PROMPT = """다음은 '{topic}'에 관한 회의 대화의 일부입니다. 이 대화 내용을 바탕으로 핵심 내용을 간결하게 요약해 주십시오.
주요 키워드: {keywords}

대화 내용:
{chunk}

위 대화 내용의 핵심 요약:"""

# --- GPT 텍스트 요약 프롬프트 (최종 통합) ---
GPT_FINAL_SUMMARY_SYSTEM_PROMPT = "You are a helpful assistant that synthesizes multiple summaries into one final, coherent summary."
GPT_FINAL_SUMMARY_USER_PROMPT = """다음은 '{topic}'에 관한 회의의 각 부분에 대한 요약들입니다. 이 요약들을 바탕으로 전체 회의 내용을 포괄하는 하나의 최종 요약문을 자연스러운 문장으로 작성해 주십시오.

개별 요약 내용:
{combined_summary_text}

최종 통합 요약문:"""

# --- Gemini 텍스트 요약 프롬프트 ---
GEMINI_SUMMARY_PROMPT_TEMPLATE = '''
당신은 회의 내용을 분석하고 핵심만 요약하는 전문 AI 어시스턴트입니다.
회의 주제: "{topic}"
주요 키워드: {keywords}

다음은 회의 전체 대화 내용입니다.
이 회의의 핵심 내용을 구조화하여 명확하고 간결하게 요약해주세요.

전체 대화 내용:
{text}

회의 요약:
'''

# --- STT 프롬프트 템플릿 ---
STT_PROMPT_TEMPLATE = "이 대화는 '{topic}'에 관한 것입니다. 주요 용어는 다음과 같습니다: {keywords}."
