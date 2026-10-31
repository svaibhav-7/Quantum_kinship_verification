# Final comparison tables

CSV files in this folder contain the full-precision comparison fields; the rendered tables below use those same rows.

## 1. Scalar versus vector readout

| Backbone | Corpus | Data status | Historical Δ | Composite Δ | 95% CI low | 95% CI high | p | Holm, 16 | BH, 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FaceNet | FIW | historical-only | 0.119860 | 0.119860 | 0.076652 | 0.163068 | 0.001529 | 0.022938 | 0.007512 |
| FaceNet | KinFaceW-I | historical-only | 0.165757 | 0.165757 | 0.121190 | 0.210325 | 0.000496 | 0.007940 | 0.007512 |
| FaceNet | KinFaceW-II | historical-only | 0.096450 | 0.096450 | 0.011520 | 0.181380 | 0.034413 | 0.240894 | 0.055062 |
| FaceNet | TSKinFace | corrected | 0.193220 | 0.075360 | 0.027763 | 0.122956 | 0.011728 | 0.105551 | 0.023456 |
| ArcFace | FIW | historical-only | 0.159649 | 0.159649 | 0.078661 | 0.240636 | 0.005423 | 0.054232 | 0.012396 |
| ArcFace | KinFaceW-I | historical-only | 0.102427 | 0.102427 | 0.063440 | 0.141414 | 0.001878 | 0.025982 | 0.007512 |
| ArcFace | KinFaceW-II | historical-only | 0.041945 | 0.041945 | 0.021806 | 0.062084 | 0.004443 | 0.048870 | 0.011847 |
| ArcFace | TSKinFace | corrected | 0.156485 | 0.061035 | -0.005798 | 0.127869 | 0.064282 | 0.385694 | 0.093502 |

## 2. Corrected amplitude VQC versus width-matched controls

| Backbone | Corpus | Control | Quantum AUC | Control AUC | Δ | 95% CI low | 95% CI high | p | Holm, 32 | BH, 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FaceNet | FIW | ctl-width-linear | 0.633587 | 0.590663 | 0.042924 | -0.054954 | 0.140802 | 0.290289 | 1.000000 | 0.344046 |
| FaceNet | FIW | ctl-width-mlp | 0.633587 | 0.663590 | -0.030004 | -0.115284 | 0.055277 | 0.383969 | 1.000000 | 0.417258 |
| FaceNet | FIW | ctl-width-random | 0.633587 | 0.504921 | 0.128666 | 0.094518 | 0.162813 | 0.000472 | 0.014154 | 0.003970 |
| FaceNet | KinFaceW-I | ctl-width-linear | 0.704869 | 0.538025 | 0.166844 | 0.131548 | 0.202140 | 0.000195 | 0.006034 | 0.003114 |
| FaceNet | KinFaceW-I | ctl-width-mlp | 0.704869 | 0.561577 | 0.143292 | 0.069433 | 0.217151 | 0.005744 | 0.114874 | 0.014138 |
| FaceNet | KinFaceW-I | ctl-width-random | 0.704869 | 0.536090 | 0.168778 | 0.137848 | 0.199708 | 0.000111 | 0.003541 | 0.003114 |
| FaceNet | KinFaceW-II | ctl-width-linear | 0.616115 | 0.499745 | 0.116370 | 0.034777 | 0.197963 | 0.016680 | 0.216842 | 0.026688 |
| FaceNet | KinFaceW-II | ctl-width-mlp | 0.616115 | 0.528535 | 0.087580 | -0.010400 | 0.185560 | 0.068084 | 0.707106 | 0.094725 |
| FaceNet | KinFaceW-II | ctl-width-random | 0.616115 | 0.497235 | 0.118880 | 0.041363 | 0.196397 | 0.013075 | 0.206040 | 0.023244 |
| FaceNet | TSKinFace | ctl-width-linear | 0.672370 | 0.628581 | 0.043789 | 0.024218 | 0.063360 | 0.003417 | 0.078596 | 0.010935 |
| FaceNet | TSKinFace | ctl-width-mlp | 0.672370 | 0.630207 | 0.042163 | 0.029984 | 0.054341 | 0.000655 | 0.018337 | 0.004191 |
| FaceNet | TSKinFace | ctl-width-random | 0.672370 | 0.623509 | 0.048861 | 0.028091 | 0.069632 | 0.002839 | 0.068132 | 0.010094 |
| ArcFace | FIW | ctl-width-linear | 0.679458 | 0.593718 | 0.085740 | 0.028243 | 0.143238 | 0.014374 | 0.206040 | 0.024209 |
| ArcFace | FIW | ctl-width-mlp | 0.679458 | 0.664427 | 0.015032 | -0.028418 | 0.058481 | 0.391179 | 1.000000 | 0.417258 |
| ArcFace | FIW | ctl-width-random | 0.679458 | 0.504756 | 0.174702 | 0.117591 | 0.231813 | 0.001054 | 0.028453 | 0.005620 |
| ArcFace | KinFaceW-I | ctl-width-linear | 0.637502 | 0.528177 | 0.109325 | 0.050204 | 0.168446 | 0.006819 | 0.129560 | 0.015586 |
| ArcFace | KinFaceW-I | ctl-width-mlp | 0.637502 | 0.518546 | 0.118956 | 0.047404 | 0.190508 | 0.009912 | 0.178414 | 0.021145 |
| ArcFace | KinFaceW-I | ctl-width-random | 0.637502 | 0.536253 | 0.101249 | 0.035523 | 0.166974 | 0.012878 | 0.206040 | 0.023244 |
| ArcFace | KinFaceW-II | ctl-width-linear | 0.531752 | 0.500370 | 0.031383 | -0.011467 | 0.074232 | 0.111776 | 0.894207 | 0.143073 |
| ArcFace | KinFaceW-II | ctl-width-mlp | 0.531752 | 0.495675 | 0.036077 | -0.016272 | 0.088427 | 0.128236 | 0.897650 | 0.157829 |
| ArcFace | KinFaceW-II | ctl-width-random | 0.531752 | 0.498285 | 0.033468 | -0.006503 | 0.073438 | 0.080716 | 0.726442 | 0.107621 |
| ArcFace | TSKinFace | ctl-width-linear | 0.659176 | 0.647275 | 0.011901 | -0.036890 | 0.060692 | 0.535412 | 1.000000 | 0.552683 |
| ArcFace | TSKinFace | ctl-width-mlp | 0.659176 | 0.652253 | 0.006924 | -0.031380 | 0.045227 | 0.642127 | 1.000000 | 0.642127 |
| ArcFace | TSKinFace | ctl-width-random | 0.659176 | 0.642032 | 0.017144 | -0.027687 | 0.061976 | 0.348190 | 1.000000 | 0.397931 |

## 3. Historical versus corrected TSKinFace

| Evaluation | Backbone | Measure | Historical | Corrected | Corrected − historical | 95% CI low | 95% CI high | Paired p | Pairing limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grouped k-fold baseline | FaceNet | roc_auc | 0.881441 | 0.798230 | -0.083211 | -0.174015 | 0.007594 | 0.063693 | Fold-index pairing; historical fold manifest unavailable. |
| Grouped k-fold baseline | FaceNet | accuracy | 79.245093 | 73.975225 | -5.269868 | -12.904939 | 2.365202 | 0.127807 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: amp-expectation | FaceNet | ROC-AUC | 0.737280 | 0.672370 | -0.064910 | -0.108145 | -0.021675 | 0.014050 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: amp-fidelity | FaceNet | ROC-AUC | 0.544060 | 0.597010 | 0.052951 | -0.006657 | 0.112559 | 0.069217 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: control | FaceNet | ROC-AUC | 0.550669 | 0.623575 | 0.072906 | 0.017216 | 0.128597 | 0.022067 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: ctl-width-linear | FaceNet | ROC-AUC | 0.514178 | 0.628581 | 0.114403 | 0.087640 | 0.141167 | 0.000289 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: ctl-width-mlp | FaceNet | ROC-AUC | 0.699193 | 0.630207 | -0.068986 | -0.199070 | 0.061099 | 0.214899 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: ctl-width-random | FaceNet | ROC-AUC | 0.531545 | 0.623509 | 0.091964 | 0.070917 | 0.113010 | 0.000265 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: amp-expectation | ArcFace | ROC-AUC | 0.668222 | 0.659176 | -0.009046 | -0.120205 | 0.102113 | 0.832319 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: amp-fidelity | ArcFace | ROC-AUC | 0.511737 | 0.598141 | 0.086404 | -0.003137 | 0.175945 | 0.055276 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: control | ArcFace | ROC-AUC | 0.509666 | 0.635141 | 0.125475 | 0.093460 | 0.157490 | 0.000405 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: ctl-width-linear | ArcFace | ROC-AUC | 0.507621 | 0.647275 | 0.139654 | 0.103534 | 0.175774 | 0.000427 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: ctl-width-mlp | ArcFace | ROC-AUC | 0.501144 | 0.652253 | 0.151108 | 0.123394 | 0.178823 | 0.000111 | Fold-index pairing; historical fold manifest unavailable. |
| Amplitude VQC: ctl-width-random | ArcFace | ROC-AUC | 0.511919 | 0.642032 | 0.130113 | 0.096387 | 0.163838 | 0.000430 | Fold-index pairing; historical fold manifest unavailable. |
| Backbone comparison: single | FaceNet | ROC-AUC | 0.732595 | 0.667538 | -0.065057 | NA | NA | NA | Not calculable: artifacts store means/SD, not fold scores. |
| Backbone comparison: set | FaceNet | ROC-AUC | 0.732595 | 0.667538 | -0.065057 | NA | NA | NA | Not calculable: artifacts store means/SD, not fold scores. |
| Backbone comparison: set+density | FaceNet | ROC-AUC | 0.732722 | 0.667345 | -0.065377 | NA | NA | NA | Not calculable: artifacts store means/SD, not fold scores. |
| Backbone comparison: single | ArcFace | ROC-AUC | 0.846043 | 0.756829 | -0.089214 | NA | NA | NA | Not calculable: artifacts store means/SD, not fold scores. |
| Backbone comparison: set | ArcFace | ROC-AUC | 0.846043 | 0.756829 | -0.089214 | NA | NA | NA | Not calculable: artifacts store means/SD, not fold scores. |
| Backbone comparison: set+density | ArcFace | ROC-AUC | 0.844677 | 0.757001 | -0.087676 | NA | NA | NA | Not calculable: artifacts store means/SD, not fold scores. |
| Set-level evaluation: single | FaceNet | ROC-AUC | 0.734359 | 0.667538 | -0.066821 | NA | NA | NA | Not calculable: summary-only artifact. Historical mean set size was 169.03 despite one TSKinFace image per identity. |
| Set-level evaluation: set | FaceNet | ROC-AUC | 0.761284 | 0.667538 | -0.093746 | NA | NA | NA | Not calculable: summary-only artifact. Historical mean set size was 169.03 despite one TSKinFace image per identity. |
| Set-level evaluation: set+density | FaceNet | ROC-AUC | 0.761284 | 0.667345 | -0.093939 | NA | NA | NA | Not calculable: summary-only artifact. Historical mean set size was 169.03 despite one TSKinFace image per identity. |

## 4. Standard VQC versus classical baselines

| Contrast | Arm A AUC | Arm B AUC | Δ | 95% CI low | 95% CI high | p | Inference note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VQC expectation minus capacity-matched VQC control | 0.646164 | 0.684372 | -0.038208 | -0.078483 | 0.002067 | 0.057946 | Same held-out folds and VQC protocol. |
| VQC fidelity minus capacity-matched VQC control | 0.530092 | 0.684372 | -0.154280 | -0.219398 | -0.089161 | 0.002765 | Same held-out folds and VQC protocol. |
| VQC expectation minus VQC fidelity | 0.646164 | 0.530092 | 0.116072 | 0.035700 | 0.196443 | 0.016000 | Readout contrast on matched held-out folds. |
| VQC expectation minus global grouped-k-fold baseline | 0.646164 | 0.798230 | -0.152066 | -0.240459 | -0.063674 | 0.008798 | Matched test folds; baseline trained on pooled corpora. |
| VQC fidelity minus global grouped-k-fold baseline | 0.530092 | 0.798230 | -0.268138 | -0.350734 | -0.185542 | 0.000839 | Matched test folds; baseline trained on pooled corpora. |

## 5. Corrected quantum-inspired methods versus classical controls

| Backbone | Method | Classical control | Method AUC | Control AUC | Δ | 95% CI low | 95% CI high | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FaceNet | amp-expectation | control | 0.672370 | 0.623575 | 0.048795 | -0.004522 | 0.102112 | 0.063917 |
| FaceNet | amp-expectation | ctl-width-linear | 0.672370 | 0.628581 | 0.043789 | 0.024218 | 0.063360 | 0.003417 |
| FaceNet | amp-expectation | ctl-width-mlp | 0.672370 | 0.630207 | 0.042163 | 0.029984 | 0.054341 | 0.000655 |
| FaceNet | amp-expectation | ctl-width-random | 0.672370 | 0.623509 | 0.048861 | 0.028091 | 0.069632 | 0.002839 |
| FaceNet | amp-fidelity | control | 0.597010 | 0.623575 | -0.026565 | -0.110182 | 0.057052 | 0.427557 |
| FaceNet | amp-fidelity | ctl-width-linear | 0.597010 | 0.628581 | -0.031571 | -0.061251 | -0.001891 | 0.041831 |
| FaceNet | amp-fidelity | ctl-width-mlp | 0.597010 | 0.630207 | -0.033197 | -0.082461 | 0.016067 | 0.134688 |
| FaceNet | amp-fidelity | ctl-width-random | 0.597010 | 0.623509 | -0.026498 | -0.054644 | 0.001647 | 0.059172 |
| ArcFace | amp-expectation | control | 0.659176 | 0.635141 | 0.024035 | -0.017543 | 0.065613 | 0.183763 |
| ArcFace | amp-expectation | ctl-width-linear | 0.659176 | 0.647275 | 0.011901 | -0.036890 | 0.060692 | 0.535412 |
| ArcFace | amp-expectation | ctl-width-mlp | 0.659176 | 0.652253 | 0.006924 | -0.031380 | 0.045227 | 0.642127 |
| ArcFace | amp-expectation | ctl-width-random | 0.659176 | 0.642032 | 0.017144 | -0.027687 | 0.061976 | 0.348190 |
| ArcFace | amp-fidelity | control | 0.598141 | 0.635141 | -0.037000 | -0.112213 | 0.038213 | 0.243746 |
| ArcFace | amp-fidelity | ctl-width-linear | 0.598141 | 0.647275 | -0.049134 | -0.121704 | 0.023435 | 0.133307 |
| ArcFace | amp-fidelity | ctl-width-mlp | 0.598141 | 0.652253 | -0.054112 | -0.119489 | 0.011266 | 0.083121 |
| ArcFace | amp-fidelity | ctl-width-random | 0.598141 | 0.642032 | -0.043891 | -0.122448 | 0.034666 | 0.195788 |
| FaceNet | standard vqc-expectation | capacity-matched VQC control | 0.646164 | 0.684372 | -0.038208 | -0.078483 | 0.002067 | 0.057946 |
| FaceNet | standard vqc-fidelity | capacity-matched VQC control | 0.530092 | 0.684372 | -0.154280 | -0.219398 | -0.089161 | 0.002765 |
| FaceNet | set+density | set-level features | 0.667345 | 0.667538 | -0.000193 | NA | NA | NA |
| ArcFace | set+density | set-level features | 0.757001 | 0.756829 | 0.000172 | NA | NA | NA |

## 6. Set-level aggregation

| Backbone | Corpus | Status | Historical single | Historical set | Historical gain | Corrected single | Corrected set | Corrected gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FaceNet | FIW | rerun; non-TSKinFace context | 0.749100 | 0.825399 | 0.076299 | 0.749100 | 0.825399 | 0.076299 |
| FaceNet | KinFaceW-I | rerun; non-TSKinFace context | 0.758756 | 0.758756 | 0.000000 | 0.758756 | 0.758756 | 0.000000 |
| FaceNet | KinFaceW-II | rerun; non-TSKinFace context | 0.777005 | 0.777005 | 0.000000 | 0.777005 | 0.777005 | 0.000000 |
| FaceNet | TSKinFace | corrected TSKinFace | 0.732595 | 0.732595 | 0.000000 | 0.667538 | 0.667538 | 0.000000 |
| ArcFace | FIW | rerun; non-TSKinFace context | 0.776399 | 0.873963 | 0.097565 | 0.776399 | 0.873963 | 0.097565 |
| ArcFace | KinFaceW-I | rerun; non-TSKinFace context | 0.754003 | 0.754003 | 0.000000 | 0.754003 | 0.754003 | 0.000000 |
| ArcFace | KinFaceW-II | rerun; non-TSKinFace context | 0.818990 | 0.818990 | 0.000000 | 0.818990 | 0.818990 | 0.000000 |
| ArcFace | TSKinFace | corrected TSKinFace | 0.846043 | 0.846043 | 0.000000 | 0.756829 | 0.756829 | 0.000000 |

## 7. Dataset-wise and backbone-wise performance

| Corpus | Status | Historical baseline AUC | Rerun baseline AUC | FaceNet single | FaceNet set | ArcFace single | ArcFace set |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FIW | rerun_on_corrected_runner | 0.810136 | 0.749455 | 0.749100 | 0.825399 | 0.776399 | 0.873963 |
| KinFaceW-I | rerun_on_corrected_runner | 0.875214 | 0.798821 | 0.758756 | 0.758756 | 0.754003 | 0.754003 |
| KinFaceW-II | rerun_on_corrected_runner | 0.831970 | 0.761325 | 0.777005 | 0.777005 | 0.818990 | 0.818990 |
| TSKinFace | corrected | 0.881441 | 0.798230 | 0.667538 | 0.667538 | 0.756829 | 0.756829 |

## 8. Historical formulation coverage

| # | Method | Classical control | Historical report | Corrected TSKinFace status | Corrected result |
| --- | --- | --- | --- | --- | --- |
| 1 | SWAP fidelity (5 seeds) | classical head | p=0.945 | Not rerun | No corrected TSKinFace result artifact |
| 2 | SWAP fidelity (20 folds) | classical head | p=0.982 | Not rerun | No corrected TSKinFace result artifact |
| 3 | Density fidelity | mean-pooling | -8.7 AUC | Not rerun | No corrected TSKinFace result artifact |
| 4 | Interference phase sweep | convex mixture | +0.1 AUC | Not rerun | No corrected TSKinFace result artifact |
| 5 | Entanglement entropy | cosine | 0.514 alone | Not rerun | No corrected TSKinFace result artifact |
| 6 | POVM head | capacity-matched | -5.5 AUC | Not rerun | No corrected TSKinFace result artifact |
| 7 | Purity regulariser | decorrelation | -0.9 AUC; p=0.047 | Not rerun | No corrected TSKinFace result artifact |
| 8 | Density fidelity (sets) | set features | At most 0.0006 AUC change | Rerun available | FaceNet delta=-0.000193; ArcFace delta=+0.000172 (summary means; fold-level p unavailable) |
| 9 | Phase sweep (triads) | convex mixture | -0.0006 AUC; p=0.147 | Not rerun | No corrected TSKinFace result artifact |
