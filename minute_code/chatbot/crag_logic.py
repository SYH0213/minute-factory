
import os
import uuid
import logging
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from .chatbot_prompts import (
    CHATBOT_GRADER_SYSTEM_PROMPT,
    CHATBOT_GRADE_PROMPT_TEMPLATE,
    CHATBOT_REWRITER_SYSTEM_PROMPT,
    CHATBOT_REWRITE_PROMPT_TEMPLATE
)

# --- 1. 기본 설정 ---
load_dotenv()

# 프로젝트 루트를 기준으로 chroma_db 경로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PERSIST_DIR = os.path.join(ROOT_DIR, "chroma_db")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# LangChain 모델 및 도구 초기화 (앱 전체에서 재사용)
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
web_search_tool = TavilySearch(max_results=3)
embeddings = OpenAIEmbeddings()

# --- 2. 벡터 저장소 (ChromaDB) 관련 함수 ---

def get_chroma_retriever(collection_name: str):
    """
    지정된 collection에 대한 ChromaDB retriever를 반환합니다.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever()

def update_vector_store(file_path: str, collection_name: str):
    """
    주어진 텍스트 파일로 ChromaDB 컬렉션을 생성하거나 업데이트합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 1. 문서 로드
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    # 2. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. ChromaDB에 문서 저장
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Vector store updated for {file_path} in collection '{collection_name}'.")


# --- 3. CRAG 그래프 정의 ---

# 가. 그래프 상태
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    collection_name: str
    web_search_needed: bool

# 나. 관련성 평가 모델
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="문서가 질문과 관련 있으면 'yes', 없으면 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments, method="function_calling")
system_grader = CHATBOT_GRADER_SYSTEM_PROMPT
grade_prompt = ChatPromptTemplate.from_messages(
    [ ("system", system_grader), ("human", CHATBOT_GRADE_PROMPT_TEMPLATE) ]
)
retrieval_grader = grade_prompt | structured_llm_grader

# 다. 질문 재작성 모델
system_rewriter = CHATBOT_REWRITER_SYSTEM_PROMPT
re_write_prompt = ChatPromptTemplate.from_messages(
    [ ("system", system_rewriter), ("human", CHATBOT_REWRITE_PROMPT_TEMPLATE) ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

# 라. 답변 생성 체인
RAG_PROMPT_KOREAN = """당신은 질문에 답변하는 AI 어시스턴트입니다. 다음 검색된 컨텍스트를 사용하여 질문에 답하세요. 만약 답을 모른다면, 모른다고 말해주세요.

질문: {question}

컨텍스트:
{context}

답변:"""
prompt_generator = ChatPromptTemplate.from_template(RAG_PROMPT_KOREAN)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = prompt_generator | llm | StrOutputParser()


# 마. 그래프 노드 함수
def retrieve(state: GraphState):
    print("---RETRIEVE---")
    question = state["question"]
    collection_name = state["collection_name"]
    retriever = get_chroma_retriever(collection_name)
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "collection_name": collection_name}

def grade_documents(state: GraphState):
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
    
    web_search_needed = not bool(filtered_docs)
    if web_search_needed:
        print("---GRADE: NO RELEVANT DOCUMENTS, INITIATE WEB SEARCH---")
        
    return {"documents": filtered_docs, "web_search_needed": web_search_needed}

def rewrite_question(state: GraphState):
    print("---REWRITE QUESTION---")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question}

def web_search(state: GraphState):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n\n".join(docs)
    web_results_doc = Document(page_content=web_results)
    documents.append(web_results_doc)

    return {"documents": documents}

def generate(state: GraphState):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}

# 바. 조건부 엣지
def decide_to_generate(state: GraphState):
    print("---ASSESS GRADED DOCUMENTS---")
    if state.get("web_search_needed", False):
        print("---DECISION: REWRITE QUESTION and WEB SEARCH---")
        return "rewrite_question"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

# --- 4. CRAG 실행 함수 ---

_app = None

def get_crag_app():
    global _app
    if _app is None:
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("rewrite_question", rewrite_question)
        workflow.add_node("web_search", web_search)
        workflow.add_node("generate", generate)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {"rewrite_question": "rewrite_question", "generate": "generate"},
        )
        workflow.add_edge("rewrite_question", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        
        _app = workflow.compile()
    return _app

def run_query(question: str, collection_name: str):
    """
    CRAG 파이프라인을 실행하여 질문에 대한 답변을 반환합니다.
    """
    if not all([os.getenv("OPENAI_API_KEY"), os.getenv("TAVILY_API_KEY")]):
        return "오류: OPENAI_API_KEY와 TAVILY_API_KEY가 .env 파일에 설정되어야 합니다."
    
    try:
        app = get_crag_app()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        inputs = {"question": question, "collection_name": collection_name}
        
        final_generation = ""
        for output in app.stream(inputs, config=config):
            if "generation" in output.get(next(reversed(output)), {}):
                final_generation = output[next(reversed(output))]['generation']
                
        return final_generation if final_generation else "답변을 생성하지 못했습니다."
    except Exception as e:
        logging.error(f"CRAG 파이프라인 실행 중 오류 발생: {e}")
        return f"챗봇 응답 생성 중 오류가 발생했습니다: {e}"
