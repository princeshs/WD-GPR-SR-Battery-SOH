#!/usr/bin/env python3
"""
Update /home/prince/Desktop/ML_Project/project_poster.pptx:
- Keep the header (Shape 0, 1, 2) completely untouched.
- Update all text boxes and content to reflect the WD-GPR-SR battery project.
- Replace Picture 24 (Architecture), Picture 34 (Table 1), and Picture 36 (Results Plot)
  with high-res, verified project figures.
"""

import shutil
import pptx
from pptx.util import Pt
from pptx.enum.text import PP_ALIGN

PPTX_PATH = "project_poster.pptx"
BACKUP_PATH = "project_poster_backup.pptx"

# 1. Create a safe backup
shutil.copyfile(PPTX_PATH, BACKUP_PATH)
print(f"Created backup at {BACKUP_PATH}")

prs = pptx.Presentation(PPTX_PATH)
slide = prs.slides[0]

def set_text_content(shape, paragraphs_data, font_name="Times New Roman", default_size=11):
    tf = shape.text_frame
    tf.word_wrap = True
    # Clear existing paragraphs except first
    p0 = tf.paragraphs[0]
    p0.text = ""
    
    # Remove extra paragraphs if any
    p_elements = tf._txBody.xpath('./a:p')
    for pe in p_elements[1:]:
        tf._txBody.remove(pe)
        
    for idx, (p_text, p_size, is_bold) in enumerate(paragraphs_data):
        if idx == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
            
        p.text = p_text
        p.font.name = font_name
        p.font.size = Pt(p_size if p_size else default_size)
        p.font.bold = is_bold
        p.alignment = PP_ALIGN.LEFT

def replace_image(shape, image_path):
    with open(image_path, "rb") as f:
        new_blob = f.read()
    rel_id = shape._element.xpath('.//a:blip/@r:embed')[0]
    image_part = slide.part.related_part(rel_id)
    image_part._blob = new_blob
    print(f"Replaced {shape.name} with {image_path} ({len(new_blob)} bytes)")

# --- 2. Update Content of Left Column ---
# Shape 5: Abstract Content
abstract_text = (
    "This work proposes a physics-guided machine learning framework for battery State-of-Health (SOH) prognostics across distinct chemistries (LiCoO₂ and LiNiMnCoO₂). Benchmarked on 4,657 cycles from NASA Ames and CALCE CS2 repositories under strict zero-leakage Leave-One-Cell-Out Cross-Validation (LOCO-CV), standard deep models suffer severe cross-cell collapse. We formulate Wide-and-Deep Sparse Variational Gaussian Process Regression with Uncertainty-Weighted Symbolic Residual Correction (WD-GPR-SR), combining linear trend tracking, deep manifold feature extraction, and parsimonious genetic programming on honest out-of-fold residuals. The proposed model achieves 1.64% RMSE on CALCE with 94.5% 95-PICP coverage and executes in 0.15 ms on NVIDIA Jetson Orin Nano with zero PyTorch runtime dependencies."
)
set_text_content(slide.shapes[5], [(abstract_text, 10.5, False)], default_size=10.5)

# Shape 7: Introduction Content
intro_bullets = [
    ("❑ Lithium-ion battery SOH prognostics is vital for EV safety, operational reliability, and thermal runaway prevention [1].", 10.5, False),
    ("❑ Prevalent literature relies on random cycle splits, suffering from severe cross-cycle autocorrelation leakage and reporting illusory R² > 0.99 that completely collapses on unseen battery cells [2].", 10.5, False),
    ("❑ Standard black-box neural networks lack Bayesian uncertainty calibration, while exact Gaussian Process Regression incurs prohibitive O(N³) computational complexity, preventing continuous real-time execution on embedded BMS edge hardware.", 10.5, False)
]
set_text_content(slide.shapes[7], intro_bullets, default_size=10.5)

# Shape 9: Problem Statement Content
prob_text = (
    "Conventional battery prognostics lacks reliable uncertainty quantification, suffers from fatal cross-cell data leakage, and exhibits the near-zero residual trap in hybrid models. We formulate a zero-leakage, physics-informed framework delivering calibrated 95% epistemic confidence intervals, human-verifiable closed-form residual equations, and sub-millisecond edge execution on NVIDIA Jetson Orin Nano."
)
set_text_content(slide.shapes[9], [(prob_text, 10.5, False)], default_size=10.5)

# --- 3. Update Content of Center Column ---
# Shape 12: Key Contributions Content
contrib_bullets = [
    ("❑ Formulated Wide-and-Deep Sparse Variational GPR (WD-SVGP) combining linear trend, ARD-RBF dynamics, and deep Matérn-5/2 manifold kernels (M=64 inducing points), cutting complexity from O(N³) to O(M²N).", 9.5, False),
    ("❑ Developed Uncertainty-Weighted Symbolic Regression (UW-SR) on honest inner LOCO out-of-fold residuals, resolving the near-zero residual trap without test data leakage.", 9.5, False),
    ("❑ Discovered an interpretable 51-node closed-form mathematical equation correcting non-linear CV charge saturation and IC peak overpotential shifts near end-of-life.", 9.5, False),
    ("❑ Deployed a standalone zero-PyTorch C engine on NVIDIA Jetson Orin Nano achieving 0.15 ms latency and 6,395 inferences/s.", 9.5, False)
]
set_text_content(slide.shapes[12], contrib_bullets, default_size=9.5)

# Shape 13: Picture 24 (Architecture diagram)
replace_image(slide.shapes[13], "results/figures/poster_architecture.png")

# Shape 14: Figure 1 Caption
slide.shapes[14].text_frame.text = "Figure 1: End to end Architecture of WD-GPR-SR"
for p in slide.shapes[14].text_frame.paragraphs:
    p.font.name = "Times New Roman"
    p.font.size = Pt(11)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

# Shape 16: Algorithm 1 Content
algo_lines = [
    ("Algorithm 1: Zero-Leakage WD-GPR-SR Prognostics", 9.0, True),
    ("Require: Dataset D = {(xᵢ, yᵢ)}, i = 1..N across cells C = {c₁, c₂, c₃, c₄}; inducing points M = 64; parsimony λ = 0.001.", 8.5, False),
    ("Ensure: Calibrated predictor ŷ(x) and 95% confidence interval [ŷ ± 1.96 · s · σ_GP] on held-out cell c_test.", 8.5, False),
    ("1. Partition cells: set C_train = C \\ {c_test}.", 8.5, False),
    ("2. Inner LOCO-CV: for each training cell k in C_train, train inner WD-SVGP on C_train \\ {k}; predict out-of-fold mean μᵢ^OOF and variance (σᵢ^OOF)².", 8.5, False),
    ("3. Compute honest residuals rᵢ^OOF = yᵢ − μᵢ^OOF and inverse-variance weights wᵢ = clip(1 / [(σᵢ^OOF)² + 10⁻⁶], q₁, q₉₉).", 8.5, False),
    ("4. Symbolic Evolution: evolve parsimonious formula g*(x) minimizing weighted MSE + λ · Nodes(g); select g* via inner CV gate.", 8.5, False),
    ("5. Fit full WD-SVGP model on C_train using compound kernel k_WD; calibrate variance scaling factor s.", 8.5, False),
    ("6. Deploy standalone engine: evaluate ŷ = μ_GP(x) + g*(x) on unseen test cell c_test.", 8.5, False)
]
set_text_content(slide.shapes[16], algo_lines, default_size=8.5)

# --- 4. Update Content of Right Column ---
# Shape 18: Picture 34 (Table 1 image)
replace_image(slide.shapes[18], "results/figures/poster_table1.png")

# Shape 19: Picture 36 (Results panel image)
replace_image(slide.shapes[19], "results/figures/poster_results_panel.png")

# Shape 20: Table 1 Caption
slide.shapes[20].text_frame.text = "Table 1: Proposed model results compare to baseline"
for p in slide.shapes[20].text_frame.paragraphs:
    p.font.name = "Times New Roman"
    p.font.size = Pt(11)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

# Shape 22: Conclusion Content
conc_text = (
    "The proposed WD-GPR-SR model establishes a zero-leakage benchmark for battery prognostics. By integrating wide linear trends, deep manifold features, and uncertainty-weighted symbolic corrections, it achieves superior accuracy (1.64% RMSE on CALCE), reliable 95% uncertainty calibration (94.5% PICP), and sub-millisecond edge execution (0.15 ms) on NVIDIA Jetson Orin Nano."
)
set_text_content(slide.shapes[22], [(conc_text, 10.0, False)], default_size=10.0)

# --- 5. Update Footer References ---
# Shape 23: References Content
refs_lines = [
    ("[1] Saha, B., & Goebel, K. (2007). Battery Data Set. NASA Ames Prognostics Data Repository, Moffett Field, CA.", 9.5, False),
    ("[2] He, W., Williard, N., Osterman, M., & Pecht, M. (2011). Prognostics of lithium-ion batteries based on Dempster-Shafer theory. IEEE Trans. Ind. Electron., 58(9), 4398–4405. | [3] Hensman, J., et al. (2013). Gaussian Processes for Big Data. Proc. UAI. | [4] Wilson, A. G., et al. (2016). Deep Kernel Learning. Proc. AISTATS.", 9.5, False)
]
set_text_content(slide.shapes[23], refs_lines, default_size=9.5)

prs.save(PPTX_PATH)
print(f"Successfully updated {PPTX_PATH} with zero header modifications!")
