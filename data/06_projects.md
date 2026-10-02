# Projects

## Ask Saurav — this RAG chatbot
The assistant you're talking to is itself one of my Generative AI projects. It is a retrieval-augmented generation (RAG) chatbot that answers questions about me in my own voice.
- **Pipeline:** my resume, papers and project notes are written as Markdown, split by heading and then into overlapping chunks with LangChain text splitters, embedded with the sentence-transformers model all-MiniLM-L6-v2 (run as ONNX through fastembed) (the same SBERT family I used in SCRBM and SecSDAE), and stored in a persistent ChromaDB vector database.
- **Retrieval and generation:** for each question, a LangChain chain rewrites follow-up questions into standalone ones using chat history, retrieves the most relevant chunks with maximal marginal relevance (MMR), and sends them to an open-weight LLM served by Groq (GPT-OSS-120B), which streams the answer token by token.
- **Grounding:** the system prompt only allows answers from retrieved context, and every answer shows its sources.
- **Backend:** FastAPI with Server-Sent Events streaming, per-IP rate limiting and CORS restricted to my portfolio. Every interaction (question, latency, sources) is logged to SQLite, and an analytics endpoint runs SQL aggregations over those logs.
- **Quality:** a pytest suite, a retrieval evaluation set measuring hit-rate@k and MRR, a notebook explaining tokenization, embeddings, chunk-size choice and ChromaDB vs FAISS, and GitHub Actions CI.
- **Hosting:** completely free. The backend is a Docker container on Render's free tier, redeployed automatically on every push to GitHub, and the chat widget is embedded in my GitHub Pages portfolio. To fit the 512 MB free container I run the embedding model in ONNX via fastembed instead of PyTorch.
- Repo: https://github.com/Saurav2021/ask-saurav

## ML Employee Performance Prediction — REST API
Nov – Dec 2025. A Random Forest + XGBoost ensemble that reaches 92% accuracy with 5-fold cross-validation, served through a Flask REST API. Each prediction comes back with a plain-language explanation of its top drivers from SHAP TreeExplainer, and the API is covered by tests. Tech: Python, Pandas, Scikit-learn, XGBoost, SHAP, Flask. Repo: https://github.com/Saurav2021/Employee-performance-predictor

## REST API Data Pipeline — Enterprise Automation
2026. An automated ETL pipeline that pulls data from REST APIs, validates it, and stores it in a relational database without duplicates using indexed, idempotent upserts and MD5-based deduplication. It also generates daily reports. Retry logic with exponential backoff, HTTP error classification and structured audit logging make failures easy to trace. Tech: Python, Requests, SQL, SQLite, logging. Repo: https://github.com/Saurav2021/REST-API-Data-Pipeline

## Mobile Sales Analytics — Power BI Dashboard
A star-schema data model with DAX KPI measures and 8 cross-filtered visuals, including geo maps, funnel charts and trend lines. Tech: Power BI, DAX, Power Query. Repo: https://github.com/Saurav2021/Mobile-Sales-Analytics-Power-BI-Dashboard

## Stock Price Prediction
Benchmarks Linear Regression, Random Forest and an LSTM on historical Yahoo Finance data with financial indicators; the LSTM performed best on MSE/RMSE. Tech: Python, TensorFlow, Pandas. Repo: https://github.com/Saurav2021/stocks_price_prediction

## Research Portfolio Website
A hand-built, responsive portfolio in HTML5, CSS3 and JavaScript presenting my research and projects. It now hosts this chatbot: https://saurav2021.github.io/Portfolio/

## Earlier projects
- Jarvis A.I.: a personal desktop voice assistant. https://github.com/Saurav2021/Jarvis-A.I.
- Face recognition system with OpenCV. https://github.com/Saurav2021/Face_recog_1
- MP3 player in Python using pygame, with play, pause and stop controls, a file-selection UI and smooth asynchronous playback. https://github.com/Saurav2021/MP3-Player
- An Operations Intelligence Dashboard, built as a take-home assignment for an AI Product Engineer role: a full web app with inventory management, order tracking, SLA-breach prediction and Chart.js analytics.
