# 유튜브 강의 요약 및 Q&A 웹 애플리케이션

유튜브 강의 영상의 자막을 가져와 OpenAI API를 활용하여 요약하고, 요약 내용을 기반으로 질문에 답변해주는 웹 애플리케이션입니다.

## 주요 기능

- 📥 **자막 추출**: 유튜브 강의 영상의 한국어/영어 자막을 자동으로 가져옵니다
- 📊 **요약 생성**: OpenAI API를 사용하여 강의 내용을 체계적으로 요약합니다
- ✨ **내용 보충**: 요약이 부실한 경우, 원본 자막을 참고하여 관련 개념과 예시를 추가로 보충합니다
- ❓ **Q&A**: 요약된 내용을 기반으로 질문에 답변합니다
- 🕸️ **Knowledge Graph**: 강의 내용에서 핵심 개념과 관계를 추출하여 지식 그래프를 생성합니다
- 📈 **3D 시각화**: 생성된 Knowledge Graph를 3D 인터랙티브 그래프로 시각화합니다

## 설치 방법

### 1. 저장소 클론 또는 다운로드

```bash
git clone <repository-url>
cd yt_summarizer
```

### 2. 가상 환경 생성 및 활성화 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env.example` 파일을 참고하여 `.env` 파일을 생성하고 OpenAI API 키를 설정하세요.

```bash
# .env 파일 생성
OPENAI_API_KEY=your_openai_api_key_here
```

OpenAI API 키는 [OpenAI Platform](https://platform.openai.com/api-keys)에서 발급받을 수 있습니다.

## 실행 방법

```bash
streamlit run app.py
```

브라우저에서 자동으로 애플리케이션이 열립니다. 기본 주소는 `http://localhost:8501`입니다.

## 사용 방법

1. **자막 가져오기**
   - 왼쪽 사이드바에 유튜브 영상 URL을 입력합니다
   - "자막 가져오기" 버튼을 클릭합니다
   - 자막이 성공적으로 가져와지면 메인 화면에 표시됩니다

2. **요약 생성**
   - "📊 요약 생성" 버튼을 클릭하여 강의 내용을 요약합니다
   - 요약이 생성되면 메인 화면에 표시됩니다

3. **내용 보충** (선택사항)
   - 요약이 부족하다고 생각되면 "✨ 내용 보충" 버튼을 클릭합니다
   - 원본 자막을 참고하여 더 상세한 요약이 생성됩니다

4. **Q&A**
   - 오른쪽 사이드바의 Q&A 섹션에서 질문을 입력합니다
   - "답변 받기" 버튼을 클릭하여 요약 내용 기반 답변을 받습니다

5. **Knowledge Graph 생성**
   - 요약이 생성된 후 "📊 Knowledge Graph 생성" 버튼을 클릭합니다
   - OpenAI API가 강의 내용에서 핵심 개념(엔티티)과 관계를 추출합니다
   - 생성된 그래프는 3D 인터랙티브 그래프로 시각화됩니다
   - 노드 크기와 색상은 연결 수(degree)를 나타냅니다

## 기술 스택

- **Streamlit**: 웹 애플리케이션 프레임워크
- **youtube-transcript-api**: 유튜브 자막 추출
- **OpenAI API**: 요약, Q&A, Knowledge Graph 생성
- **python-dotenv**: 환경 변수 관리
- **NetworkX**: 그래프 구조 생성 및 분석
- **Plotly**: 3D 인터랙티브 그래프 시각화
- **NumPy**: 수치 연산

## 주의사항

- 유튜브 영상에 자막이 활성화되어 있어야 합니다
- 한국어 또는 영어 자막이 있는 영상만 지원합니다
- OpenAI API 사용 시 비용이 발생할 수 있습니다
- API 키는 절대 공개 저장소에 업로드하지 마세요

## 라이선스

MIT License

