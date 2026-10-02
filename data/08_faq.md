# Frequently Asked Questions

## Tell me about yourself (elevator pitch)
I'm Saurav Kumar, an AI engineer who just completed a B.Tech in Computer Science (AI) at GEC Munger, ranked 1st in my department with a CGPA of 8.79. In production, I worked on a live 200+ camera CCTV system at L&T Smart City, where my adaptive frame-sampling fix cut CPU load by 35% with zero dropped feeds, and at Twinverse I built the REST API layer between Python ML services and a React app (40% faster responses) plus a BERT keyword extractor (28% more accurate). In research, I'm first author on three papers: AURA, an explainable four-stage video pipeline for women's safety (YOLOv11, tracking, R(2+1)D-18, SHAP); SCRBM, a Sentence-BERT-conditioned RBM recommender that beats LightGCN; and SecSDAE, a sequential recommender with gated identity-text fusion. I also built this RAG chatbot with LangChain, ChromaDB and Groq. I'm looking for Generative AI and AI/ML engineering roles and can relocate anywhere in India.

## Why should we hire you for a Generative AI role?
I combine three things a fresher GenAI engineer needs. First, real NLP and embeddings depth: I have used Sentence-BERT embeddings, contrastive learning and Transformer blocks in two first-author research papers, and shipped a BERT-based feature in a production API. Second, I build working systems end to end: this RAG chatbot uses LangChain, ChromaDB, a Groq-hosted LLM, FastAPI, SQLite and CI, and it is deployed for free. Third, production discipline: I supported a live 200+ camera system under SLAs at L&T Smart City and cut its CPU load by 35%. I also measure things honestly; my papers report failures as openly as wins.

## What is your strongest skill?
Python-based machine learning engineering, from data and model to API to deployment. In GenAI specifically, retrieval and embeddings, because my recommender-systems research is built on the same semantic-embedding ideas that power RAG.

## Are you open to relocation? When can you join?
Yes. I'm based in Patna and open to relocating anywhere in India, including Noida, Bengaluru, Hyderabad and Pune. I'm a 2026 graduate and can join at short notice.

## What kind of team do you want to join?
A team building real AI products, especially LLM applications, RAG systems, intelligent chatbots and AI data pipelines, where I can learn from senior engineers and ship things people use.

## Have you participated in hackathons, Kaggle or open source?
I run all my research experiments on Kaggle GPUs (T4 and P100) and design them around Kaggle's 12-hour session limits. My projects, including this chatbot, are public on GitHub. For anything more specific, email me.

## How do you handle a problem you don't know how to solve?
I read the primary source, build the smallest version that works, and measure it. Most of my research skill came this way: learning RBMs, contrastive learning and video models from papers and reproducing baselines myself under tight compute limits.

## What are your salary expectations or notice period?
I'd prefer to discuss compensation directly. Please email me at sauravsuz@gmail.com. As a fresher I can join immediately.

## How does this chatbot work?
It is a retrieval-augmented generation system I built. Your question is embedded with all-MiniLM-L6-v2, matched against chunks of my resume and papers in ChromaDB, and the best chunks are passed to an LLM on Groq with strict instructions to answer only from them, in my voice. The sources appear under each answer. The code is open: https://github.com/Saurav2021/ask-saurav

## Can I see your resume?
Yes. Email me at sauravsuz@gmail.com and I'll send my latest resume right away. You can also browse my work at https://saurav2021.github.io/Portfolio/ and https://github.com/Saurav2021.
