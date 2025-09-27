# -*- coding: utf-8 -*-
import os
import logging
import concurrent.futures
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 클라이언트 및 체인 초기화 --- #

def get_openai_client():
    """OpenAI 클라이언트를 생성하고 반환합니다."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return None
    return OpenAI(api_key=api_key)

def get_gemini_chain(template_string, input_vars):
    """LangChain과 Gemini를 사용하는 LLMChain을 생성합니다."""
    if not os.getenv("GOOGLE_API_KEY"):
        logging.error("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return None
    
    prompt = PromptTemplate(template=template_string, input_variables=input_vars)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    return LLMChain(llm=llm, prompt=prompt)

# --- 텍스트 교정 (Correction) --- #

def _correct_with_gpt(client, text, topic, keywords):
    """GPT-4를 사용하여 텍스트를 교정합니다."""
    prompt = f"""다음 텍스트는 '{topic}'에 대한 회의 내용입니다. 
    주요 키워드는 {', '.join(keywords)} 입니다. 
    문맥에 맞게 문장을 다듬고, 맞춤법 및 띄어쓰기를 수정해주세요. 
    특히, 키워드가 포함된 문장은 더 자연스럽게 만들어주세요.

    원본 텍스트:
    {text}

    교정된 텍스트:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects and refines meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"GPT-4 교정 중 오류 발생: {e}")
        return text

def _correct_with_gemini(text, topic, keywords):
    """Gemini Pro를 사용하여 텍스트를 교정합니다."""
    template = '''
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
    chain = get_gemini_chain(template, ["text", "topic", "keywords"])
    if not chain: return text
    try:
        return chain.run(text=text, topic=topic, keywords=", ".join(keywords))
    except Exception as e:
        logging.error(f"Gemini 교정 중 오류 발생: {e}")
        return text

def correct_text(llm_choice, text, topic, keywords):
    """선택된 LLM을 사용하여 텍스트를 교정합니다."""
    logging.info(f"LLM({llm_choice})으로 텍스트 교정을 시작합니다...")
    if llm_choice == "gpt-4":
        client = get_openai_client()
        if not client: return text
        return _correct_with_gpt(client, text, topic, keywords)
    elif llm_choice == "gemini-pro":
        return _correct_with_gemini(text, topic, keywords)
    else:
        logging.warning(f"지원하지 않는 LLM 모델({llm_choice})입니다. 원본 텍스트를 반환합니다.")
        return text

# --- 텍스트 요약 (Summarization) --- #

def _summarize_with_gpt_mapreduce(client, text, topic, keywords):
    """GPT-4-turbo와 MapReduce 방식으로 긴 텍스트를 요약합니다."""
    
    def get_summary_for_chunk(chunk):
        prompt = f"""다음은 '{topic}'에 관한 회의 대화의 일부입니다. 이 대화 내용을 바탕으로 핵심 내용을 간결하게 요약해 주십시오.
        주요 키워드: {', '.join(keywords)}

        대화 내용:
        {chunk}

        위 대화 내용의 핵심 요약:"""
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes parts of a meeting transcript."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"GPT-4 개별 요약 API 호출 중 오류: {e}")
            return ""

    if len(text) < 12000:
        logging.info("텍스트가 짧아 직접 요약을 실행합니다.")
        return get_summary_for_chunk(text)

    logging.info("텍스트가 길어 '맵리듀스' 방식으로 요약을 실행합니다.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        intermediate_summaries = list(executor.map(get_summary_for_chunk, chunks))
    
    combined_summary_text = "\n\n---\n\n".join(filter(None, intermediate_summaries))
    
    final_summary_prompt = f"""다음은 '{topic}'에 관한 회의의 각 부분에 대한 요약들입니다. 이 요약들을 바탕으로 전체 회의 내용을 포괄하는 하나의 최종 요약문을 자연스러운 문장으로 작성해 주십시오.

    개별 요약 내용:
    {combined_summary_text}

    최종 통합 요약문:"""
    try:
        final_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that synthesizes multiple summaries into one final, coherent summary."},
                {"role": "user", "content": final_summary_prompt}
            ],
            temperature=0.7,
        )
        return final_response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"GPT-4 최종 요약 중 오류 발생: {e}")
        return "최종 요약 생성에 실패했습니다."

def _summarize_with_gemini(text, topic, keywords):
    """Gemini Pro를 사용하여 텍스트를 요약합니다."""
    template = '''
    당신은 회의 내용을 분석하고 핵심만 요약하는 전문 AI 어시스턴트입니다.
    회의 주제: "{topic}"
    주요 키워드: {keywords}

    다음은 회의 전체 대화 내용입니다.
    이 회의의 핵심 내용을 구조화하여 명확하고 간결하게 요약해주세요.

    전체 대화 내용:
    {text}

    회의 요약:
    '''
    chain = get_gemini_chain(template, ["text", "topic", "keywords"])
    if not chain: return "요약 생성 실패"
    try:
        return chain.run(text=text, topic=topic, keywords=", ".join(keywords))
    except Exception as e:
        logging.error(f"Gemini 요약 중 오류 발생: {e}")
        return "요약 생성에 실패했습니다."

def summarize_text(llm_choice, text, topic, keywords):
    """선택된 LLM을 사용하여 텍스트를 요약합니다."""
    logging.info(f"LLM({llm_choice})으로 텍스트 요약을 시작합니다...")
    if llm_choice == "gpt-4":
        client = get_openai_client()
        if not client: return "요약 생성 실패"
        return _summarize_with_gpt_mapreduce(client, text, topic, keywords)
    elif llm_choice == "gemini-pro":
        return _summarize_with_gemini(text, topic, keywords)
    else:
        logging.warning(f"지원하지 않는 LLM 모델({llm_choice})입니다. 요약을 건너뜁니다.")
        return "지원하지 않는 모델입니다."
