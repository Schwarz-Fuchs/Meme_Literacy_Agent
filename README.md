# Meme Literacy Agent: Intent-Driven Meme Retrieval & Multi-Agent Alignment

> **核心主张**：真正的 Meme 检索不是“把用户原话盲目塞进搜索引擎”。本系统构建了一个**认知智能 Agent 架构**，通过多 Agent 深度思考解构用户的幽默机制与情绪隐喻，并引入多模态视觉对齐（VLM-in-the-loop）解决传统文本匹配“文不对图”的痛点。

## 🚀 快速运行

#### 1.配置运行环境

```powershell
pip install -r requirements.txt
```
#### 2.编辑 .env，填入你的 API key；也可暂时跳过，程序会使用规则回退
[点击查看设置](.env)

#### 3.激活环境后启动 app
```powershell
streamlit run app.py
```

Windows 若阻止 Activate.ps1，可只在当前窗口执行 `Set-ExecutionPolicy -Scope Process Bypass`，或改用 `.venv\Scripts\python -m streamlit run app.py`。

## Demo 图结构

```text
START → understand → search → rank → reflect → END
          LLM 意图      DDG     基线排序   展示/追问
```

## 核心攻坚方向

### 1. 攻坚 Con 1：图文语义对齐（LLM 懂用户情绪，但如何让系统懂 Meme？）
* **痛点**：原始基线仅依赖文本标题与关键词重合度计算，容易导致“文字对得上，图片牛头不对马嘴”。
* **可能解决方案**：
  * **VLM-in-the-Loop 视觉校验层**：引入视觉多模态模型作为“梗图鉴赏裁判”。将候选图片的画面与用户的 `MemeIntent`（情绪、视觉隐喻）进行联合输入，利用 VLM 动态打分并过滤低质量匹配。
  * **CLIP/SigLIP 向量空间对齐**：将用户的抽象意图文本与候选图片统一编码至多模态嵌入空间，计算图文语义相似度。
  * **TransMeme: A Multi-Agent Framework for Cross-Cultural Meme Transcreation**：参考一下相关文章。

### 2. 攻坚 Con 2：多 Agent 协作与深度思考（参考 deepseek-R1 模式）
* **痛点**：demo 的单 Agent 试图同时处理深层语境解构、梗文化联想和精准英文检索式转换，效果肯定不佳。
* **可能解决方案**：将 `understand` 节点拆解为多角色对抗与协同集群：
---

## 可探索路线

1. **多源异构检索层**：
   * 突破单一的 DuckDuckGo 限制，接入 **Tenor API / Giphy API** 或周期性抓取结构化标签。
2. **知识库与持久化缓存（RAG）**：
   * 引入向量数据库（如 Chroma / FAISS）存储高频 Meme 映射与历史偏好，避免对相同/相似输入重复调用 LLM 带来的延迟和成本。
3. **Human-in-the-Loop 与偏好反馈（RLHF）**：
   * 通过前端交互（如“太贴切 / 模板对但图不对”）收集用户的隐式反馈，将历史偏好写入数据库，用于后续微调检索权重。
4. **动态热梗同步机制**：
   * 针对网络梗“时效性强”的特点，设计后台任务定期同步主流模因网站（如 Know Your Meme）的最新词条，突破大模型的知识盲区。

---

## 核心工程警示

- **LLM 不天然“懂梗”**：大模型提供的仅是假设生成与结构化解释，其正确性必须通过候选池、多模态视觉对齐以及用户交互闭环来验证。
- **透明度优先**：当前的元数据基线与中间 Trace 日志均保持完全可解释，增加模块时请一同记录trace，便于在学术评估中定位误差来源。
- **生产环境考量**：万一真实上线，需额外考虑版权合规、NSFW/仇恨内容过滤、来源白名单、API 速率限制及系统缓存。