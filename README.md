# Meme Literacy Agent: Intent-Driven Meme Retrieval & Multi-Agent Alignment

> **核心主张**：真正的 Meme 检索不是“把用户原话盲目塞进搜索引擎”。本系统构建了一个**基于LangGraph 的 AI-Agent 架构**，意图通过多 Agent 深度思考解构用户的幽默机制与情绪隐喻，并引入多模态视觉对齐（VLM-in-the-loop）解决传统文本匹配“文不对图”的痛点。

## 🚀 快速运行

#### 1. 克隆项目并创建虚拟环境
```powershell
conda create -n [env_name] python=3.11 -y
conda activate [env_name]
```
#### 2. 安装依赖
```powershell
pip install -r requirements.txt
```
#### 3.编辑 .env文件，填入你的 API key；也可暂时跳过，程序会使用规则回退
[点击查看设置](.env)

#### 4.激活环境后启动 app (请先cd到app.py所在目录)
```powershell
streamlit run app.py
```

Windows 若阻止 Activate.ps1，可只在当前窗口执行 `Set-ExecutionPolicy -Scope Process Bypass`，或改用 `.venv\Scripts\python -m streamlit run app.py`。

## Demo 图结构

```text
START → understand → search → rank → reflect → END
          LLM 意图      DDG     基线排序   展示/追问
```

## ⚠️ 项目状态 

**本仓库目前仅处于启动阶段**

### 好消息
1. **端到端可运行**：在本地环境及 DeepSeek API 支持下，能够跑起来。
2. **结构化与可解释**：利用 Pydantic 定义了 `MemeIntent` 契约，并通过中间 Trace 日志保持全流程透明。

### 坏消息
1. **“懂用户” vs “懂 Meme” 的对齐鸿沟 (Alignment Gap)**
   * **现状**：LLM 分析文本语境和情绪是可行的，但将其直接映射并对齐到视觉 Meme 图上效果有限。换句话说：**LLM 懂用户，但它真的懂 Meme 吗？**
   * **可能解法**：引入 **RAG 机制**，以 *Know Your Meme* 等权威梗图数据库作为外挂知识库；或者引入 **VLM-in-the-Loop** 进行视觉校验。
2. **多义性与用户偏好 (User Preference & Ambiguity)**
   * **现状**：用户的输入往往具有多义性，合适的 Meme 候选可能有很多，且不同社交语境下的偏好完全不同。
   * **可能解法**：由于难以直接用 LLM 自我评分（容易陷入“它是否真的懂梗”的裁判悖论），系统急需引入 **Human-in-the-Loop（人在回路）** 的显式/隐式偏好反馈。
3. **缺乏多 Agent 协作与深度思考 (Lack of Agent Collaboration)**
   * **现状**：目前仅由单一 Agent 盲目直接输出 `image_search_queries` 和情绪分析，缺乏分工。
   * **可能解法**：参考类似 **DeepSeek Chain of Thought (CoT)** 的深度思考模式，或多 Agent 分工机制（如参考论文 [_TransMeme: A Multi-Agent Framework for Cross-Cultural Meme Transcreation_](https://arxiv.org/abs/2608.27127) 这样具备跨文化、多角色协作的框架）。
4. **单一搜索引擎瓶颈**
   * **现状**：仅依赖 DuckDuckGo 免费搜索。
   * **可能解法**：未来应扩展为多源异构检索，增加专门抓取表情包或结构化标签的专用搜索引擎/API。
5. **缺少持久化存储与语义缓存 (Lack of Database & Caching)**
   * **现状**：数据全在内存和临时 JSON 中。
   * **可能解法**：引入轻量级数据库，用于对高频/相似输入做**语义缓存 (Semantic Cache)**，避免每次都重复调用昂贵的 LLM API。
6. **缺乏客观验证指标 (Evaluation Paradox)**
   * **现状**：没有设置客观的评测指标。让 LLM/VLM 自己当裁判打分会陷入“裁判本身不懂梗”的悖论，最终的校验权仍需回归人类用户。

## 核心攻坚方向
1. **攻坚 Con 1 & 2：图文语义，用户偏好对齐**
2. **攻坚 Con 3：多 Agent 协作与深度思考**

---
## 核心工程警示
- **LLM 不天然“懂梗”**：大模型提供的仅是假设生成与结构化解释，其正确性必须通过候选池、多模态视觉对齐以及用户交互闭环来验证。
- **透明度优先**：当前的元数据基线与中间 Trace 日志均保持完全可解释，增加模块时请一同记录trace，便于在学术评估中定位误差来源。
- **生产环境考量**：万一真实上线，需额外考虑版权合规、NSFW/仇恨内容过滤、来源白名单、API 速率限制及系统缓存。