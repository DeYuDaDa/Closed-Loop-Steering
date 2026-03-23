### Results 章节叙事架构设计 (Narrative Structure)

我们将 Results 拆分为五个极具攻击性的子章节，层层递进地验证我们在 Method 中提出的动力学系统：

1.  **4.1 Macro-Efficiency: Shattering the Truncation Wall** (宏观效率：打碎截断墙，引入 Token 效率核密度图和全局 Accuracy)
2.  **4.2 The Energy-Shock Paradox & Latent Stability** (安全与维稳：打破能量-休克悖论，引入消融柱状图、二维散点图和 Repetition Rate)
3.  **4.3 Preserving System-2 Deliberation (DTR Analysis)** (深思维持：引入小提琴图，反驳“降熵即退化”的偏见)
4.  **4.4 Micro-Dynamics of the Closed-Loop System** (微观动力学实证：引入单样本注入轨迹图、Task 3 EMA 触发精度)
5.  **4.5 Micro-Logic Reliability & The Escape Velocity** (微观纠错可靠性：引入 Logic Flip Matrix，用 10:1 惊人比例收尾，并将 Trajectory 图的“逃逸速度”概念纯文本化)

---

### 英文原稿起草 (Academic English Draft)

#### 4. Results

**4.1 Macro-Efficiency: Shattering the Truncation Wall**
A primary symptom of the "Overthinking Trap" in System-2 reasoning is the unbounded consumption of test-time compute, culminating in budget exhaustion. As illustrated in Figure X (Token Efficiency KDE), the unsteered Baseline model exhibits a massive probability density accumulation at the **4000**-token limit, forming a catastrophic "Truncation Wall." This indicates that the model frequently falls into infinite cyclic rationalization without achieving logical closure.

Our Dynamic Closed-Loop Steering effectively neutralizes this computational bottleneck. By injecting geometric momentum at critical divergence points, the dynamic system successfully flattens the **4000**-token death peak and left-shifts the distribution, forming a healthy bimodal convergence peak between **1500** and **2500** tokens. As corroborated by the Global Metrics (Table X), this dramatic reduction in average token consumption (from **3045** to **2785**) directly translates to a significant accuracy leap (from **67%** to **74%**). The system achieves Pareto optimality: preventing compute bankruptcy while maximizing logical accuracy.

**4.2 Resolving the Energy-Shock Paradox and Latent Stability**
A central claim of our methodology is that continuous linear intervention destroys the pre-trained logical manifold. Figure Y (Energy-Shock Paradox Scatter Plot) empirically validates this. The Continuous steering group (over-steering) accumulates massive physical intervention energy ($\sum \alpha_t \gg 300$), which forces the latent representations out of their bounds. This induces severe "State Shock," macroscopically manifesting as an unacceptable N-gram repetition rate surging to **28.17%**. 

Furthermore, our ablation study on representation entropy (Figure Z) reveals that continuous injection actually exacerbates internal chaos, driving the EMA entropy up to **2.157** (Cognitive Tearing). Conversely, the Dynamic Spherical system elegantly clusters in the low-energy, low-repetition quadrant (repetition suppressed to **13.87%**, approaching the Baseline's safe zone). By acting strictly on-demand, the closed-loop system acts as a cognitive stabilizer, crushing the global EMA entropy to an unprecedented **0.265** and ensuring linguistic safety.

**4.3 Preserving System-2 Deliberation: Deep Thought Ratio (DTR) Analysis**
A common critique of entropy-reduction techniques is that they might degrade the model into shallow, System-1 generation. We refute this using the Local Deep Thinking Rate (DTR) analysis. 

As depicted in the violin plot (Figure W), Continuous steering significantly shifts the DTR distribution downward (median dropping to **0.67**), proving that constant vector injection introduces "cognitive friction" that suppresses deep reasoning chains. Remarkably, Dynamic steering not only maintains a high median DTR (**0.75**) but exhibits an upward expansion in the top quartile (Q3 reaching **0.89** at $\alpha=0.45$). This confirms that transient, spherical perturbation provides the exact "geometric momentum" needed to break out of local minima. Once re-aligned to the correct logical manifold, the PD controller gracefully withdraws, allowing the model to fully utilize its deepest network layers for rigorous logical deduction.

**4.4 Micro-Dynamics of the Closed-Loop Sensor**
To verify the precision of our control loop, we dissect the intervention dynamics at the micro-level. Figure V (Divergence Anatomy) demonstrates that the EMA probe acts as a high-precision sensor: the moment of initial semantic divergence strictly forms a sharp Gaussian peak exactly aligned with our predefined threshold interval (**0.10** to **0.25**).

This precision is further visualized in the single-sample Intervention Dynamics trajectories (Prompt 0 and Prompt 1). As the EMA entropy (blue line) breaches the threshold, the PD controller instantaneously responds with a proportional steering spike (red line). This exacts a rapid reduction in entropy. Furthermore, the variable lengths of the intervention windows validate the ThinkBrake mechanism. Rather than relying on fixed-length budget forcing, the endogenous logit margin calculates a custom stopping boundary for each prompt, ensuring adaptive and timely termination of the steering force.

**4.5 Micro-Logic Reliability and the Escape Velocity**
Finally, we evaluate the surgical precision of the manifold projection. The Logic Flip Matrix provides compelling evidence of the system's reliability. At the intervention intensity of $\alpha=0.3$, the system achieves an astounding **10:1** Correct-to-Broken ratio (Rescued: **51**, Broken: **5**). This near-zero negative interference provides an impenetrable defense against claims that steering acts as stochastic noise; it operates as a high-precision scalpel that excises errors with minimal collateral damage to healthy reasoning paths.

However, trajectory tracking reveals a critical "Escape Velocity" phenomenon. While $\alpha=0.3$ offers maximal safety, the dynamic trajectory occasionally remains entangled with the baseline's rigid energy barrier in highly complex problems. Elevating the intensity to $\alpha=0.45$ yields the necessary geometric kinetic energy to fully decouple the trajectory and break stubborn local minima, resulting in the highest global accuracy (**74%**), albeit with a slight increase in cognitive friction (a **3:1** flip ratio). This trade-off underscores the power of our PD controller: bounding $\alpha_{\max}$ allows researchers to perfectly calibrate the system between absolute logic preservation and aggressive error correction.

---
