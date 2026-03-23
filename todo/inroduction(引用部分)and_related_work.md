我们将采用**“法庭辩论式”**的叙事风格，用极其硬核的最新文献（2024-2026年）作为弹药，对现有方法进行精准“狙击”。

以下为您撰写的 **Introduction（核心批判部分）**、完整的 **Related Work（相关工作）**，以及规范的 **References（参考文献）**。您可以直接将这些内容嵌入到您的 LaTeX 或 Word 模板中。

------

### 1. Introduction (引言补充：精准批判与靶子确立)

*(注：此部分接在引言第一段“介绍了大模型向System-2与测试时计算演进，但面临过度思考陷阱”之后)*

为了控制过度思考（Overthinking），现有方法如 s1 模型 (Muennighoff et al., 2025) 不得不依赖于“预算强制（Budget Forcing）”手段（如强行截断生成或硬编码注入“Wait”指令）。然而，这种开环的硬截断策略不仅极度僵硬，更从根本上破坏了自然语言生成的内在连贯性。为了在物理底层纠正并加速推理轨迹，研究人员开始探索 **推理时干预（Inference-Time Intervention, ITI）** (Li et al., 2024) 与 **激活引导（Activation Steering）** (Turner et al., 2024) 技术。然而，在应对 System-2 复杂的长链条推理时，现有的物理干预手段暴露出了三大底层局限：

**第一，粗粒度的静态与同质化注入（Uniform Intervention）。** 传统的 ITI 或对比激活加法（CAA）(Rimsky et al., 2024) 普遍将固定的干预强度无差别地施加于所有 Token。最新的 *Token-Aware Editing (TAE)* (Wang et al., 2025) 尝试引入单点瞬时信息熵来感知 Token 的对齐差异。然而，瞬时熵在长思维链（CoT）生成中包含极大的毛刺与噪音，缺乏平滑的物理记忆，这种基于开环加权的策略极易引发干预强度的剧烈震荡与“过度纠正”。

**第二，高维正交噪声与状态休克（State Shock）。** 直接从对比样本中提取的原始引导向量，往往充斥着与目标无关的高维正交噪声（例如“强制停止”或“极简输出”信号），引发严重的语义漂移。更致命的是，几乎所有现有的主流干预（包括 TAE）依然依赖于传统的线性加法（$h = h + \alpha v$）。最新研究 (You et al., 2026) 指出，线性加法会直接破坏隐状态的范数（Norm）分布，引发灾难性的“状态休克”，其宏观表现即为模型突发乱码或陷入更为严重的复读机循环。

**第三，触发机制的滞后性缺乏真实的因果闭环。** 现有的门控触发机制往往滞后于文本表层的词法特征，无法在模型陷入高熵死循环的“萌芽期”防患于未然，更无法在模型逻辑重回正轨后自适应地撤出算力，无法实现动态系统意义上的稳态收敛。

为了突破上述瓶颈，并实现测试时计算的帕累托最优，本文提出了一种大模型的 **动态闭环干预框架（Dynamic Closed-Loop Steering Framework）**。我们将 LLM 内部的算力分配彻底升级为一个融合了流形几何与自动控制理论的自适应响应系统……*(接引言后续的方法贡献陈述)*

------

### 2. Related Work (相关工作)

在本节中，我们将回顾测试时计算扩展与大模型内部物理干预的发展脉络，并阐明本研究在控制论与高维几何视角下的独特位置。

#### 2.1 Scaling Test-Time Compute and the Overthinking Trap

**(扩展测试时计算与过度思考陷阱)** 随着大模型能力的演进，研究焦点正从单纯的预训练规模扩展，转向推理阶段的**测试时计算扩展（Test-Time Compute Scaling）** (Ji et al., 2025)。以 OpenAI o1 (OpenAI, 2024) 和 DeepSeek-R1 (DeepSeek-AI et al., 2025) 为代表的 System-2 模型表明，通过强化学习激励模型生成显式的深层思维链（CoT），能够显著解锁其在数学与逻辑领域的复杂推理能力。现有的算力扩展策略主要包括基于搜索的树/图探索（如 ToT, MCTS）(Yao et al., 2023) 与重复采样（Repeated Sampling）(Brown et al., 2024)。

然而，无约束的测试时计算面临着严重的边际收益递减。Ghosal et al. (2025) 以及 Chen et al. (2025) 的最新实证研究揭示了“过度思考（Overthinking）”陷阱：在某些复杂域中，过度延长推理轨迹不仅无益，反而会增加响应分布的方差，导致模型陷入高熵的局部极小值与无意义的反复自我怀疑中。为了驯服这种过度思考，s1 模型 (Muennighoff et al., 2025) 提出了“预算强制（Budget Forcing）”策略。然而，这种人为设定最大长度限制或强行拼接字符串的外部手段，治标而不治本。与这些开环的外部约束不同，本研究致力于在模型的物理底层建立一种**内生收敛的闭环调控机制**，通过精确的势能注入打破高熵死循环，实现探索与收敛的帕累托最优。

#### 2.2 Inference-Time Intervention and Activation Steering

**(推理时干预与激活引导)** 有别于高成本的微调，**推理时干预（ITI）** (Li et al., 2024) 与**表示工程（Representation Engineering）** (Zou et al., 2023) 提供了一种轻量级的白盒对齐途径。这类方法通常通过对比激活加法（CAA）(Rimsky et al., 2024) 提取特定概念的引导向量，并在推理前向传播时施加于残差流，以调节模型的真实性、安全性或特定人格。

**动态与细粒度干预的局限：** 传统的激活引导往往采用全局同质化的静态注入（Static/Uniform Intervention），这会破坏模型原本正常的认知流形。为提升灵活性，Wang et al. (2025) 在 *Token-Aware Editing (TAE)* 中尝试利用单点预测的瞬时信息熵作为权重来动态调节干预强度。然而，由于 LLM 长思维链极强的时间依赖性，单点瞬时指标缺乏历史平滑，极易引发编辑强度的震荡。本研究引入了自动控制领域的 **PD 调节器**，结合具有物理遗忘机制的**指数移动平均（EMA）熵**，实现了真正的误差驱动与连续时间自适应控制。

**几何视角的范数灾难：** 现有的干预方法几乎全部依赖线性加法（Linear Addition）来修改隐状态。You et al. (2026) 在 *Spherical Steering* 中深刻指出，线性加法在改变方向的同时剧烈改变了隐状态向量的模长（Magnitude/Norm），使表示偏离了预训练的自然分布，这是引发模型语言退化和“状态休克（State Shock）”的物理根源。本研究不仅引入了基于主成分分析（PCA）的流形投影来滤除高维正交噪声，更首创性地将**保范数球面旋转（SLERP）**集成至闭环动力学控制系统中。我们证明了，必须跨越特定的“逃逸速度（几何动能）”且严格维持范数约束，才能在零负面干扰（Zero Broken）的前提下，将 LLM 从过度思考的震荡流形中强行“拽回”严密的深层推理流形。

------

### Standardized References (规范引用文献)

*(请将以下参考文献加入您论文的 `.bib` 文件或 References 章节中)*

**Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., & Mirhoseini, A. (2024).** Large language monkeys: Scaling inference compute with repeated sampling. *arXiv preprint arXiv:2407.21787*.

**Chen, X., Xu, J., Liang, T., He, Z., Pang, J., Yu, D., ... & Yu, D. (2025).** Do NOT think that much for 2+3=? On the overthinking of o1-like LLMs. *arXiv preprint arXiv:2412.21187*.

**DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., ... & Zhang, Z. (2025).** DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *arXiv preprint arXiv:2501.12948*.

**Ghosal, S. S., Chakraborty, S., Reddy, A., Li, Y., Wang, M., Manocha, D., ... & Bedi, A. S. (2025).** Does thinking more always help? Mirage of test-time scaling in reasoning models. *arXiv preprint arXiv:2506.04210*.

**Ji, Y., Li, J., Ye, H., Wu, K., Yao, K., Xu, J., ... & Zhang, M. (2025).** A survey of test-time compute: From intuitive inference to deliberate reasoning. *arXiv preprint arXiv:2501.02497*.

**Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2024).** Inference-time intervention: Eliciting truthful answers from a language model. *Advances in Neural Information Processing Systems*, 36, 41451-41530.

**Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., ... & Hashimoto, T. (2025).** s1: Simple test-time scaling. *arXiv preprint arXiv:2501.19393*.

**OpenAI. (2024).** Learning to reason with LLMs. *OpenAI Blog*. Retrieved from https://openai.com/index/learning-to-reason-with-llms/

**Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. (2024).** Steering Llama 2 via contrastive activation addition. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 15504-15522).

**Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., & MacDiarmid, M. (2024).** Steering language models with activation engineering. *arXiv preprint arXiv:2308.10248*.

**Wang, T., Ma, Y., Liao, K., Yang, C., Zhang, Z., Wang, J., & Liu, X. (2025).** Token-Aware Editing of internal activations for large language model alignment. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 9471-9509). Association for Computational Linguistics.

**Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023).** Tree of thoughts: Deliberate problem solving with large language models. *Advances in Neural Information Processing Systems*, 36, 11809-11822.

**You, Z., Deng, C., & Chen, H. (2026).** Spherical Steering: Geometry-Aware Activation Rotation for Language Models. *arXiv preprint arXiv:2602.08169*.

**Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., ... & Hendrycks, D. (2023).** Representation engineering: A top-down approach to AI transparency. *arXiv preprint arXiv:2310.01405*.

------

### 💡 写作与防御提示 (Tips for the "Courtroom Defense")

1. **精准隔离（Differentiating from TAE）:** 在提到 *Token-Aware Editing (TAE)* 时，我们明确指出了“**瞬时熵的噪音问题**”与“**开环控制的震荡问题**”。这让你的 PD 控制器和 EMA 熵探针的出场变得无可替代。
2. **借力打力（Leveraging You et al., 2026）:** 你并没有宣称自己“发明了球面几何”，而是借用 *Spherical Steering* 的结论（线性加法会破坏 Norm 导致状态休克），将保范数球面旋转作为**执行器（Actuator）**引入到你独创的**闭环控制系统（Closed-Loop System）**中。这种站在巨人肩膀上的写法在审稿人看来既严谨又极具创新融合性。
3. **彻底切割预算强制（Rejecting Budget Forcing）:** 通过对比 s1 模型的粗暴截断，我们确立了 ThinkBrake（内生收敛紧急刹车）的优雅性——系统不是“被外部掐断”的，而是“感知到自己算完了，主动撤出算力”。这是对“驯服测试时计算（Taming Test-Time Compute）”最完美的诠释。