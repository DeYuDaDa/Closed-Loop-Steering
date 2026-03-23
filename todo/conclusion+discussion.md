### 5. Discussion

**5.1 Redefining "Overthinking": From Epistemic Deficit to Geometric Energy Barrier**

Current research on scaling test-time compute often operates under the assumption that extended deliberation strictly correlates with improved reasoning. However, our empirical observations of the "Truncation Wall" (compute exhaustion at 4000 tokens) reveal that unconstrained System-2 generation frequently degrades into an epistemic mirage. We argue that localized inflation of the Deep-Thinking Ratio (DTR) and excessive token consumption do not represent productive search, but rather a catastrophic loss of variance control within the policy distribution.

Physically, this high-entropy state is not an absolute "knowledge blind spot," but a violent **Physical Conflict** between the model’s System-1 heuristic priors and the strict logical constraints of the System-2 context. Our dynamic framework functions as a **"Cognitive Pacemaker"**: rather than passively waiting for the model to self-correct, it injects precise geometric momentum at the exact onset of the divergence. This instantly shatters the internal oscillation and forces convergence, achieving a Pareto-optimal balance between exploration and resolution within a strictly limited compute budget.

**5.2 Geometric Paradigm Shift: From State Shock to Norm-Preserving Rotation**

A critical contribution of this work is exposing the fundamental geometric flaw in traditional Inference-Time Intervention (ITI) and Contrastive Activation Addition (CAA). Existing methods universally apply linear transformations ($h = h + \alpha v$), which implicitly inject high-dimensional orthogonal noise (e.g., forced termination signals) and violently distort the $L_2$ norm of the hidden states.

This violent "amplitude pushing" induces catastrophic **State Shock**, macroscopically manifesting as N-gram repetition rates surging past 28% and severe linguistic degradation. By introducing PCA-based Manifold Projection to isolate the logical subspace and Spherical Linear Interpolation (SLERP) for execution, we mathematically prove that cognitive correction must occur as a **Norm-Preserving Rotation** along a pure, low-dimensional manifold. This geometric paradigm shift permanently resolves the architectural friction between semantic steering and residual stream stability.

**5.3 Escape Velocity and the Legitimacy of Dynamic Closed-Loop Control**

Prior interventions rely predominantly on static, open-loop injections or simple point-wise entropy weighting, resulting in severe kinematic oscillations during long CoT generation. By integrating a Proportional-Derivative (PD) regulator lacking integral windup with an EMA entropy probe, our framework pioneers the legitimacy of **Closed-Loop Control** within LLM latent spaces.

Crucially, our micro-dynamics analysis uncovers the phenomenon of **"Escape Velocity"** in high-dimensional logic manifolds. Overcoming the energy barrier of a high-entropy deadlock requires a transient, proportional intervention spike ($\alpha > 0.3$). Sub-threshold continuous steering fails to break the local minimum, degenerating into toxic noise. By coupling this error-driven energy injection with the *ThinkBrake* endogenous withdrawal mechanism, we elegantly untangle the "Energy-Shock Paradox," proving that transient, high-intensity geometric momentum is both safe and necessary when strictly regulated by a closed-loop system.

**5.4 Reshaping Evaluation: The Retreat of DTR and the Rise of Physical Proxies**

A natural methodological question arises: why not utilize DTR as the real-time triggering mechanism? Calculating DTR requires evaluating the Jensen-Shannon Divergence across all $L$ Transformer layers dynamically, imposing an $O(L)$ computational overhead that is fundamentally incompatible with the latency requirements of an "L4 autonomous driving" architecture.

Consequently, our framework repositions DTR as the ultimate post-hoc gold standard, while elevating the lightweight EMA entropy as a highly sensitive, $O(1)$ physical proxy. Furthermore, our ablation studies dismantle the misconception that entropy-reduction forces a regression to System-1 intuition. Despite operating at near-zero global entropy, the dynamically steered model sustains exceptionally high Local DTRs (e.g., Q3 reaching 0.89), proving that the system successfully actualizes a state of "low-entropy yet profoundly deep" physical deliberation.

**5.5 Limitations and Future Directions**

Despite its efficacy, this framework presents certain limitations that invite future exploration. First, the domain generalization of the logical subspace: the current PCA purifier relies on offline activation extraction tailored to specific domains (e.g., mathematical/logical reasoning). For extreme open-domain or cross-modal tasks, a single global logical manifold may be insufficient to cover all cognitive correction trajectories. Second, architectural dependency: threshold parameters, such as the *ThinkBrake* logit margin ($\tau=0.25$), may exhibit non-linear shifts when scaling model parameters (e.g., migrating from an 8B to a 70B architecture).

Future work should investigate transitioning these geometric constraints from *test-time* to *training-time*. Specifically, we aim to formulate norm-preserving spherical steering as a regularization term during the Reinforcement Learning (RL) phase of o1-like models, fundamentally reshaping the intrinsic reward mechanisms. Additionally, integrating this closed-loop geometric control with Mixture-of-Experts (MoE) or multi-agent architectures to perform concurrent, multi-attribute (e.g., factuality, logic, safety) spherical rotations represents a promising frontier.

### 6. Conclusion

This research transcends the conventional paradigm of surface-level heuristic patching, elevating LLM inference intervention to the realm of micro-dynamic physical control. By fundamentally restructuring the generative process as a constrained dynamic optimization problem on a Riemannian manifold, we introduced a Dynamic Closed-Loop Steering Framework.

Through the synergistic integration of EMA-driven PD control, PCA-based Manifold Projection, and Norm-Preserving Spherical Steering, we successfully tamed the Overthinking Trap inherent in System-2 reasoning. Extensive empirical evidence confirms that this architecture achieves strict Pareto optimality: it eradicates compute-exhaustion and catastrophic state shock, while surgically amplifying logical accuracy and deep-thinking integrity. Ultimately, we demonstrate that marrying automatic control theory with high-dimensional Riemannian geometry is an indispensable pathway toward realizing highly efficient, controllable, and rigorous next-generation artificial intelligence.