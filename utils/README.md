# Utils

## LOSS

1. loss to train net_G:
$$
\begin{aligned}
L= & \lambda_{\text {CLIP }} L_{\text {CLIP }}+\lambda_{\text {geom }} L_{\text {geom }} \\
& +\lambda_{G A N} L_{G A N}+\lambda_{\text {cycle }} L_{\text {cycle }}
\end{aligned}
$$

2. loss to train VQA(BEC loss).
