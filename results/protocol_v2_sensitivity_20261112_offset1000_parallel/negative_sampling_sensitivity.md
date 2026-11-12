# Negative-sampling sensitivity

Descriptive variation across negative-sampling draws. Per-draw family-bootstrap CIs are not pooled; no hypothesis tests are performed.

Offsets: 1000, 2000

| Backbone | Corpus | Comparison | View | Mean Δ AUC | SD across draws | Range |
|---|---|---|---|---:|---:|---:|
| arcface | FIW | hybrid_q1 − hybrid_q0 | colour | -0.0027 | 0.0041 | [-0.0055, +0.0002] |
| arcface | FIW | hybrid_q1 − hybrid_q0 | matched | -0.0043 | 0.0032 | [-0.0066, -0.0020] |
| arcface | KinFaceW-I | hybrid_q1 − hybrid_q0 | colour | +0.0086 | 0.0040 | [+0.0058, +0.0115] |
| arcface | KinFaceW-I | hybrid_q1 − hybrid_q0 | matched | +0.0068 | 0.0207 | [-0.0078, +0.0214] |
| arcface | KinFaceW-II | hybrid_q1 − hybrid_q0 | colour | +0.0106 | 0.0086 | [+0.0045, +0.0167] |
| arcface | KinFaceW-II | hybrid_q1 − hybrid_q0 | matched | +0.0065 | 0.0155 | [-0.0044, +0.0175] |
| arcface | TSKinFace | hybrid_q1 − hybrid_q0 | colour | +0.0246 | 0.0002 | [+0.0245, +0.0248] |
| arcface | TSKinFace | hybrid_q1 − hybrid_q0 | matched | +0.0120 | 0.0085 | [+0.0060, +0.0180] |
| facenet | FIW | hybrid_q1 − hybrid_q0 | colour | +0.0048 | 0.0009 | [+0.0042, +0.0055] |
| facenet | FIW | hybrid_q1 − hybrid_q0 | matched | +0.0036 | 0.0002 | [+0.0034, +0.0037] |
| facenet | KinFaceW-I | hybrid_q1 − hybrid_q0 | colour | -0.0028 | 0.0002 | [-0.0029, -0.0027] |
| facenet | KinFaceW-I | hybrid_q1 − hybrid_q0 | matched | -0.0010 | 0.0050 | [-0.0045, +0.0026] |
| facenet | KinFaceW-II | hybrid_q1 − hybrid_q0 | colour | +0.0012 | 0.0092 | [-0.0053, +0.0077] |
| facenet | KinFaceW-II | hybrid_q1 − hybrid_q0 | matched | -0.0004 | 0.0033 | [-0.0027, +0.0019] |
| facenet | TSKinFace | hybrid_q1 − hybrid_q0 | colour | -0.0056 | 0.0031 | [-0.0078, -0.0034] |
| facenet | TSKinFace | hybrid_q1 − hybrid_q0 | matched | -0.0036 | 0.0001 | [-0.0036, -0.0036] |
| arcface | FIW | fair − cosine | colour | -0.0113 | 0.0013 | [-0.0122, -0.0104] |
| arcface | FIW | fair − cosine | matched | -0.0098 | 0.0008 | [-0.0104, -0.0092] |
| arcface | FIW | fair_colour_trained − cosine | colour | -0.0178 | 0.0035 | [-0.0203, -0.0154] |
| arcface | FIW | fair_colour_trained − cosine | matched | -0.0187 | 0.0044 | [-0.0218, -0.0156] |
| arcface | FIW | hybrid_paper_q1 − cosine | colour | -0.1368 | 0.0072 | [-0.1419, -0.1317] |
| arcface | FIW | hybrid_paper_q1 − cosine | matched | -0.1364 | 0.0056 | [-0.1404, -0.1324] |
| arcface | FIW | lr − cosine | colour | -0.0292 | 0.0057 | [-0.0332, -0.0251] |
| arcface | FIW | lr − cosine | matched | -0.0269 | 0.0042 | [-0.0299, -0.0240] |
| arcface | KinFaceW-I | fair − cosine | colour | +0.0137 | 0.0072 | [+0.0087, +0.0188] |
| arcface | KinFaceW-I | fair − cosine | matched | +0.0120 | 0.0020 | [+0.0105, +0.0134] |
| arcface | KinFaceW-I | fair_colour_trained − cosine | colour | +0.0114 | 0.0007 | [+0.0108, +0.0119] |
| arcface | KinFaceW-I | fair_colour_trained − cosine | matched | +0.0115 | 0.0027 | [+0.0096, +0.0135] |
| arcface | KinFaceW-I | hybrid_paper_q1 − cosine | colour | -0.1746 | 0.0019 | [-0.1759, -0.1733] |
| arcface | KinFaceW-I | hybrid_paper_q1 − cosine | matched | -0.1579 | 0.0116 | [-0.1661, -0.1497] |
| arcface | KinFaceW-I | lr − cosine | colour | +0.0218 | 0.0021 | [+0.0204, +0.0233] |
| arcface | KinFaceW-I | lr − cosine | matched | +0.0196 | 0.0015 | [+0.0186, +0.0207] |
| arcface | KinFaceW-II | fair − cosine | colour | -0.0210 | 0.0016 | [-0.0222, -0.0199] |
| arcface | KinFaceW-II | fair − cosine | matched | -0.0230 | 0.0014 | [-0.0240, -0.0220] |
| arcface | KinFaceW-II | fair_colour_trained − cosine | colour | -0.0281 | 0.0017 | [-0.0292, -0.0269] |
| arcface | KinFaceW-II | fair_colour_trained − cosine | matched | -0.0274 | 0.0063 | [-0.0318, -0.0230] |
| arcface | KinFaceW-II | hybrid_paper_q1 − cosine | colour | -0.2855 | 0.0087 | [-0.2917, -0.2793] |
| arcface | KinFaceW-II | hybrid_paper_q1 − cosine | matched | -0.2806 | 0.0074 | [-0.2858, -0.2753] |
| arcface | KinFaceW-II | lr − cosine | colour | -0.0524 | 0.0064 | [-0.0569, -0.0479] |
| arcface | KinFaceW-II | lr − cosine | matched | -0.0537 | 0.0001 | [-0.0537, -0.0537] |
| arcface | TSKinFace | fair − cosine | colour | -0.0100 | 0.0093 | [-0.0166, -0.0035] |
| arcface | TSKinFace | fair − cosine | matched | -0.0116 | 0.0083 | [-0.0174, -0.0057] |
| arcface | TSKinFace | fair_colour_trained − cosine | colour | -0.0108 | 0.0033 | [-0.0131, -0.0085] |
| arcface | TSKinFace | fair_colour_trained − cosine | matched | -0.0094 | 0.0085 | [-0.0154, -0.0034] |
| arcface | TSKinFace | hybrid_paper_q1 − cosine | colour | -0.3042 | 0.0134 | [-0.3137, -0.2948] |
| arcface | TSKinFace | hybrid_paper_q1 − cosine | matched | -0.2953 | 0.0145 | [-0.3056, -0.2851] |
| arcface | TSKinFace | lr − cosine | colour | -0.0447 | 0.0042 | [-0.0476, -0.0417] |
| arcface | TSKinFace | lr − cosine | matched | -0.0477 | 0.0045 | [-0.0509, -0.0445] |
| facenet | FIW | fair − cosine | colour | +0.0024 | 0.0067 | [-0.0023, +0.0071] |
| facenet | FIW | fair − cosine | matched | +0.0010 | 0.0064 | [-0.0035, +0.0055] |
| facenet | FIW | fair_colour_trained − cosine | colour | +0.0021 | 0.0040 | [-0.0007, +0.0050] |
| facenet | FIW | fair_colour_trained − cosine | matched | +0.0009 | 0.0037 | [-0.0017, +0.0036] |
| facenet | FIW | hybrid_paper_q1 − cosine | colour | -0.0509 | 0.0049 | [-0.0544, -0.0475] |
| facenet | FIW | hybrid_paper_q1 − cosine | matched | -0.0516 | 0.0082 | [-0.0574, -0.0458] |
| facenet | FIW | lr − cosine | colour | -0.0031 | 0.0017 | [-0.0044, -0.0019] |
| facenet | FIW | lr − cosine | matched | -0.0044 | 0.0026 | [-0.0062, -0.0026] |
| facenet | KinFaceW-I | fair − cosine | colour | +0.0113 | 0.0039 | [+0.0086, +0.0141] |
| facenet | KinFaceW-I | fair − cosine | matched | +0.0093 | 0.0053 | [+0.0056, +0.0130] |
| facenet | KinFaceW-I | fair_colour_trained − cosine | colour | +0.0011 | 0.0021 | [-0.0004, +0.0025] |
| facenet | KinFaceW-I | fair_colour_trained − cosine | matched | +0.0035 | 0.0067 | [-0.0012, +0.0082] |
| facenet | KinFaceW-I | hybrid_paper_q1 − cosine | colour | -0.0339 | 0.0005 | [-0.0343, -0.0335] |
| facenet | KinFaceW-I | hybrid_paper_q1 − cosine | matched | -0.0158 | 0.0127 | [-0.0247, -0.0068] |
| facenet | KinFaceW-I | lr − cosine | colour | +0.0307 | 0.0023 | [+0.0291, +0.0324] |
| facenet | KinFaceW-I | lr − cosine | matched | +0.0406 | 0.0122 | [+0.0319, +0.0492] |
| facenet | KinFaceW-II | fair − cosine | colour | +0.0112 | 0.0073 | [+0.0060, +0.0163] |
| facenet | KinFaceW-II | fair − cosine | matched | +0.0058 | 0.0008 | [+0.0052, +0.0063] |
| facenet | KinFaceW-II | fair_colour_trained − cosine | colour | +0.0130 | 0.0063 | [+0.0085, +0.0174] |
| facenet | KinFaceW-II | fair_colour_trained − cosine | matched | +0.0054 | 0.0027 | [+0.0035, +0.0073] |
| facenet | KinFaceW-II | hybrid_paper_q1 − cosine | colour | -0.0501 | 0.0034 | [-0.0525, -0.0477] |
| facenet | KinFaceW-II | hybrid_paper_q1 − cosine | matched | -0.0544 | 0.0024 | [-0.0561, -0.0526] |
| facenet | KinFaceW-II | lr − cosine | colour | +0.0144 | 0.0057 | [+0.0104, +0.0184] |
| facenet | KinFaceW-II | lr − cosine | matched | +0.0045 | 0.0082 | [-0.0013, +0.0103] |
| facenet | TSKinFace | fair − cosine | colour | +0.0449 | 0.0043 | [+0.0419, +0.0479] |
| facenet | TSKinFace | fair − cosine | matched | +0.0368 | 0.0033 | [+0.0344, +0.0391] |
| facenet | TSKinFace | fair_colour_trained − cosine | colour | +0.0438 | 0.0021 | [+0.0423, +0.0453] |
| facenet | TSKinFace | fair_colour_trained − cosine | matched | +0.0312 | 0.0059 | [+0.0270, +0.0354] |
| facenet | TSKinFace | hybrid_paper_q1 − cosine | colour | +0.0202 | 0.0133 | [+0.0108, +0.0297] |
| facenet | TSKinFace | hybrid_paper_q1 − cosine | matched | +0.0228 | 0.0080 | [+0.0172, +0.0285] |
| facenet | TSKinFace | lr − cosine | colour | +0.0558 | 0.0059 | [+0.0516, +0.0599] |
| facenet | TSKinFace | lr − cosine | matched | +0.0524 | 0.0024 | [+0.0507, +0.0541] |

Protocol source SHA-256: `3e3fbdcd1683bdd31f2d793884a5d37126d2e33a0134566315b622888473115a`
