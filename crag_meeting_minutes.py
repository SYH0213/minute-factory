
import os
from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import END, StateGraph, START
from langchain_core.runnables import RunnableConfig
import uuid

# --- 1. 환경 설정 ---
load_dotenv()

# LangSmith 추적 설정 (선택 사항)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "CRAG Meeting Minutes"

# --- 2. 데이터 준비 (텍스트 파일 기반) ---

# 텍스트 로더를 사용하여 test1.txt 파일 로드
loader = TextLoader("C:\\Users\\SBA\\github\\Minute\\test1.txt", encoding='utf-8')
docs = loader.load()

# 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 벡터 저장소 생성 (FAISS 사용)
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 검색기(retriever) 생성
retriever = vectorstore.as_retriever()

# --- 3. CRAG 구성 요소 정의 ---

# 모델 정의
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# 가. 문서 관련성 평가 (Retrieval Grader)
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성을 평가하기 위한 이진 점수."""
    binary_score: str = Field(
        description="문서가 질문과 관련이 있으면 'yes', 그렇지 않으면 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_grader = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grader),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader

# 나. 답변 생성 (Generator)
prompt_generator = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt_generator | llm | StrOutputParser()

# 다. 질문 재작성 (Question Rewriter)
system_rewriter = """You a question re-writer that converts an input question to a better version that is optimized
for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewriter),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

# 라. 웹 검색 도구 (Web Search Tool)
web_search_tool = TavilySearchResults(max_results=3)


# --- 4. LangGraph 그래프 정의 ---

# 가. 그래프 상태 (Graph State)
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]

# 나. 그래프 노드 (Nodes)
def retrieve(state: GraphState):
    """
    Retrieve documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """
    Generate answer
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)

    if len(filtered_docs) == 0:
        print("---GRADE: NO RELEVANT DOCUMENTS, INITIATE WEB SEARCH---")
        web_search = "Yes"

    return {"documents": filtered_docs, "web_search": web_search, "question": question}


def web_search(state: GraphState):
    """
    Web search based on the re-written question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

def rewrite_question(state: GraphState):
    """
    Transform the query to produce a better question.
    """
    print("---REWRITE QUESTION---")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question}

# 다. 조건부 엣지 (Conditional Edge)
def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, or re-generate the question and search the web.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search == "Yes":
        print("---DECISION: REWRITE QUESTION and WEB SEARCH---")
        return "rewrite_question"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

# 라. 그래프 빌드
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("web_search", web_search)


# 엣지 설정
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite_question": "rewrite_question",
        "generate": "generate",
    },
)
workflow.add_edge("rewrite_question", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# 그래프 컴파일
app = workflow.compile()

# --- 5. 실행 ---
if __name__ == '__main__':
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 질문 설정
    inputs = {"question": "회의록 검색 및 쿼리처리 방법은?"}
    
    # 그래프 실행 및 결과 출력
    for output in app.stream(inputs, config=config):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

    # 최종 답변 출력
    final_generation = output[next(reversed(output))]['generation']
    print("\n======= 최종 답변 =======\n")
    print(final_generation)

    print("\n\n=========================================\n\n")

    # 웹 검색이 필요한 질문 예시
    inputs = {"question": "2024년 노벨문학상 수상자는 누구야?"}
    
    # 그래프 실행 및 결과 출력
    for output in app.stream(inputs, config=config):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")
        
    # 최종 답변 출력
    final_generation = output[next(reversed(output))]['generation']
    print("\n======= 최종 답변 =======\n")
    print(final_generation)
