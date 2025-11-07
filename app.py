import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import json

# 환경 변수 로드 (현재 파일의 디렉토리 기준)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# .env 파일을 직접 읽어서 환경 변수 설정 (백업 방법)
if not os.getenv("OPENAI_API_KEY") and env_path.exists():
    try:
        with open(env_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except Exception:
        pass

# OpenAI 클라이언트 초기화 (API 키가 있을 때만)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# 페이지 설정
st.set_page_config(
    page_title="유튜브 강의 요약 및 Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 유튜브 강의 요약 및 Q&A")
st.markdown("유튜브 강의 영상의 자막을 가져와 요약하고, 질문에 답변해드립니다.")

# 세션 상태 초기화
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'enhanced_summary' not in st.session_state:
    st.session_state.enhanced_summary = None
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None

def extract_video_id(url):
    """유튜브 URL에서 비디오 ID 추출"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    """유튜브 비디오의 transcript 가져오기"""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id, languages=['ko', 'en'])
        transcript_list = transcript_data.to_raw_data()
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        return transcript_text, None
    except TranscriptsDisabled:
        return None, "이 영상에는 자막이 비활성화되어 있습니다."
    except NoTranscriptFound:
        return None, "이 영상에는 한국어 또는 영어 자막을 찾을 수 없습니다."
    except Exception as e:
        return None, f"오류가 발생했습니다: {str(e)}"

def summarize_transcript(transcript):
    """OpenAI API를 사용하여 transcript 요약"""
    if not client:
        return None, "OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 교육 전문가입니다. 유튜브 강의 영상의 자막을 분석하여 핵심 내용을 체계적으로 요약해주세요. 강의의 주요 개념, 예시, 핵심 포인트를 명확하게 정리해주세요."
                },
                {
                    "role": "user",
                    "content": f"다음 강의 자막을 요약해주세요. 강의의 주요 개념, 핵심 내용, 중요한 예시를 포함하여 체계적으로 정리해주세요:\n\n{transcript}"
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"요약 생성 중 오류가 발생했습니다: {str(e)}"

def enhance_summary(summary, transcript):
    """요약이 부실한 경우 내용을 보충"""
    if not client:
        return None, "OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 교육 전문가입니다. 주어진 요약을 검토하고, 원본 자막을 참고하여 부족한 부분을 보충해주세요. 관련 개념 설명, 구체적인 예시, 추가 설명 등을 포함하여 더 완성도 높은 요약을 만들어주세요."
                },
                {
                    "role": "user",
                    "content": f"다음 요약을 검토하고, 원본 자막을 참고하여 내용을 보충해주세요:\n\n[현재 요약]\n{summary}\n\n[원본 자막 일부]\n{transcript[:3000]}"
                }
            ],
            temperature=0.7,
            max_tokens=2500
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"내용 보충 중 오류가 발생했습니다: {str(e)}"

def answer_question(question, summary):
    """요약 내용을 기반으로 질문에 답변"""
    if not client:
        return None, "OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    try:
        enhanced_context = st.session_state.enhanced_summary if st.session_state.enhanced_summary else summary
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 교육 전문가입니다. 주어진 강의 요약 내용을 기반으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요."
                },
                {
                    "role": "user",
                    "content": f"다음 강의 요약 내용을 기반으로 질문에 답변해주세요:\n\n[강의 요약]\n{enhanced_context}\n\n[질문]\n{question}"
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"답변 생성 중 오류가 발생했습니다: {str(e)}"

def extract_knowledge_graph(transcript, summary):
    """OpenAI API를 사용하여 Knowledge Graph 추출"""
    if not client:
        return None, "OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    try:
        # 요약이 있으면 요약을 사용, 없으면 transcript 일부 사용
        content = summary if summary else transcript[:5000]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 Knowledge Graph 전문가입니다. 주어진 텍스트에서 핵심 개념(엔티티)과 그들 간의 관계를 추출하여 JSON 형식으로 반환해주세요. 각 관계는 'source'(시작 개념), 'target'(끝 개념), 'relation'(관계 유형)으로 표현해주세요."
                },
                {
                    "role": "user",
                    "content": f"다음 텍스트에서 Knowledge Graph를 추출해주세요. 핵심 개념과 관계를 JSON 형식으로 반환해주세요:\n\n{content}\n\n응답 형식:\n{{\n  \"entities\": [\"개념1\", \"개념2\", ...],\n  \"relations\": [\n    {{\"source\": \"개념1\", \"target\": \"개념2\", \"relation\": \"관계유형\"}},\n    ...\n  ]\n}}"
                }
            ],
            temperature=0.7,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result, None
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 텍스트에서 추출 시도
        try:
            text = response.choices[0].message.content
            # 간단한 파싱 시도
            entities = []
            relations = []
            # 기본적인 추출 로직
            return {"entities": entities, "relations": relations}, None
        except Exception as e:
            return None, f"Knowledge Graph 파싱 중 오류가 발생했습니다: {str(e)}"
    except Exception as e:
        return None, f"Knowledge Graph 생성 중 오류가 발생했습니다: {str(e)}"

def build_networkx_graph(kg_data):
    """Knowledge Graph 데이터를 NetworkX 그래프로 변환"""
    G = nx.Graph()
    
    # 엔티티 추가
    entities = kg_data.get("entities", [])
    for entity in entities:
        G.add_node(entity)
    
    # 관계 추가
    relations = kg_data.get("relations", [])
    for rel in relations:
        source = rel.get("source", "")
        target = rel.get("target", "")
        relation_type = rel.get("relation", "관련")
        
        if source and target:
            G.add_edge(source, target, relation=relation_type)
    
    return G

def visualize_3d_graph(G):
    """NetworkX 그래프를 3D로 시각화"""
    if len(G.nodes()) == 0:
        return None
    
    # 3D 레이아웃 생성 (Spring layout을 3D로 확장)
    pos_2d = nx.spring_layout(G, k=2, iterations=50)
    
    # 2D 좌표를 3D로 변환 (z축은 랜덤 또는 degree 기반)
    pos_3d = {}
    for node in G.nodes():
        x, y = pos_2d[node]
        # z축은 노드의 연결 수(degree)에 비례
        z = G.degree(node) * 0.1
        pos_3d[node] = (x, y, z)
    
    # 엣지 좌표 추출
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # 노드 좌표 추출
    node_x = [pos_3d[node][0] for node in G.nodes()]
    node_y = [pos_3d[node][1] for node in G.nodes()]
    node_z = [pos_3d[node][2] for node in G.nodes()]
    
    # 노드 크기 (degree 기반)
    node_sizes = [G.degree(node) * 10 + 10 for node in G.nodes()]
    
    # 엣지 트레이스
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        showlegend=False
    )
    
    # 노드 트레이스
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_sizes,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="연결 수"),
            line=dict(width=2, color='white')
        ),
        text=list(G.nodes()),
        textposition="middle center",
        textfont=dict(size=10, color='black'),
        hovertext=[f"{node}<br>연결 수: {G.degree(node)}" for node in G.nodes()],
        hoverinfo='text',
        showlegend=False
    )
    
    # 3D 그래프 생성
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="Knowledge Graph (3D)",
        scene=dict(
            xaxis=dict(title="X", showbackground=False),
            yaxis=dict(title="Y", showbackground=False),
            zaxis=dict(title="Z (연결 수)", showbackground=False),
            bgcolor="white",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

# 사이드바 - URL 입력
with st.sidebar:
    st.header("📥 영상 입력")
    url = st.text_input("유튜브 URL을 입력하세요", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("자막 가져오기", type="primary"):
        if not url:
            st.error("URL을 입력해주세요.")
        else:
            video_id = extract_video_id(url)
            if not video_id:
                st.error("유효한 유튜브 URL이 아닙니다.")
            else:
                st.session_state.video_id = video_id
                with st.spinner("자막을 가져오는 중..."):
                    transcript, error = get_transcript(video_id)
                    if error:
                        st.error(error)
                        st.session_state.transcript = None
                    else:
                        st.session_state.transcript = transcript
                        st.success("자막을 성공적으로 가져왔습니다!")
                        # 요약과 보충 요약, Knowledge Graph 초기화
                        st.session_state.summary = None
                        st.session_state.enhanced_summary = None
                        st.session_state.knowledge_graph = None

# 메인 영역
if st.session_state.transcript:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 자막 내용")
        with st.expander("자막 보기", expanded=False):
            st.text_area("", st.session_state.transcript, height=200, disabled=True, label_visibility="collapsed")
        
        # 요약 생성
        if not st.session_state.summary:
            if st.button("📊 요약 생성", type="primary"):
                with st.spinner("요약을 생성하는 중..."):
                    summary, error = summarize_transcript(st.session_state.transcript)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.summary = summary
                        st.success("요약이 생성되었습니다!")
        
        # 요약 표시
        if st.session_state.summary:
            st.header("📋 요약")
            st.markdown(st.session_state.summary)
            
            # 내용 보충
            col_enhance1, col_enhance2 = st.columns([1, 4])
            with col_enhance1:
                if st.button("✨ 내용 보충", use_container_width=True):
                    with st.spinner("내용을 보충하는 중..."):
                        enhanced, error = enhance_summary(st.session_state.summary, st.session_state.transcript)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.enhanced_summary = enhanced
                            st.success("내용이 보충되었습니다!")
            
            if st.session_state.enhanced_summary:
                st.header("✨ 보충된 요약")
                st.markdown(st.session_state.enhanced_summary)
            
            # Knowledge Graph 생성
            st.divider()
            st.header("🕸️ Knowledge Graph")
            
            if not st.session_state.knowledge_graph:
                if st.button("📊 Knowledge Graph 생성", type="primary"):
                    with st.spinner("Knowledge Graph를 생성하는 중..."):
                        kg_data, error = extract_knowledge_graph(
                            st.session_state.transcript,
                            st.session_state.enhanced_summary or st.session_state.summary
                        )
                        if error:
                            st.error(error)
                        else:
                            st.session_state.knowledge_graph = kg_data
                            st.success("Knowledge Graph가 생성되었습니다!")
            
            if st.session_state.knowledge_graph:
                # Knowledge Graph 정보 표시
                entities = st.session_state.knowledge_graph.get("entities", [])
                relations = st.session_state.knowledge_graph.get("relations", [])
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("엔티티 수", len(entities))
                with col_info2:
                    st.metric("관계 수", len(relations))
                
                # NetworkX 그래프 생성 및 시각화
                G = build_networkx_graph(st.session_state.knowledge_graph)
                
                if len(G.nodes()) > 0:
                    fig = visualize_3d_graph(G)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 그래프 정보
                    with st.expander("📋 그래프 상세 정보", expanded=False):
                        st.write("**엔티티 목록:**")
                        st.write(", ".join(entities[:20]) + ("..." if len(entities) > 20 else ""))
                        
                        st.write("**관계 목록:**")
                        for i, rel in enumerate(relations[:10]):
                            st.write(f"- {rel.get('source', '')} → {rel.get('target', '')} ({rel.get('relation', '')})")
                        if len(relations) > 10:
                            st.write(f"... 외 {len(relations) - 10}개 관계")
                else:
                    st.warning("생성된 Knowledge Graph에 노드가 없습니다.")
    
    with col2:
        st.header("❓ Q&A")
        
        if st.session_state.summary:
            # 질문 입력
            question = st.text_input("질문을 입력하세요", placeholder="예: 주요 개념은 무엇인가요?")
            
            if st.button("답변 받기", type="primary", use_container_width=True):
                if question:
                    with st.spinner("답변을 생성하는 중..."):
                        answer, error = answer_question(question, st.session_state.summary)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.last_answer = answer
                            st.markdown("### 답변")
                            st.markdown(answer)
                else:
                    st.warning("질문을 입력해주세요.")
            
            # 이전 답변 표시
            if 'last_answer' in st.session_state:
                st.markdown("### 최근 답변")
                st.markdown(st.session_state.last_answer)
        else:
            st.info("먼저 요약을 생성해주세요.")

else:
    st.info("👈 왼쪽 사이드바에서 유튜브 URL을 입력하고 자막을 가져오세요.")
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. `.env` 파일을 생성하고 API 키를 설정해주세요.")

