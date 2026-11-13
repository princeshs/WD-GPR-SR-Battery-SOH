#!/usr/bin/env python3
"""
Generate a comprehensive, publication-quality academic poster in draw.io XML format.
Matches the layout, typography, and styling of the reference poster sample:
- Header: IIT Patna emblem, Title, Author/Supervisor, IMPACT Lab logo
- Left Column: Abstract, Introduction, Problem Statement
- Center Column (Detailed Architecture): Key Contributions, Figure 1 (End-to-End WD-GPR-SR), Algorithm 1
- Right Column: Result and Discussion (Table 1 Benchmark), Discovered Equation, Conclusion
- Bottom Footer: References
"""

import os
import zlib
import base64
import urllib.parse
import html

class DrawioPosterBuilder:
    def __init__(self, width=2240, height=1280):
        self.width = width
        self.height = height
        self.cells = []
        self.cell_id = 1

    def next_id(self):
        self.cell_id += 1
        return str(self.cell_id)

    def add_cell(self, value, style, x, y, w, h, parent="1", is_vertex=True):
        cid = self.next_id()
        val_esc = html.escape(value, quote=True)
        # Convert newlines to HTML breaks if value contains html formatting
        if "<p>" not in value and "<div>" not in value and "\n" in value:
            val_esc = val_esc.replace("\n", "&#xa;")
            
        cell_xml = f'<mxCell id="{cid}" value="{val_esc}" style="{style}" vertex="{1 if is_vertex else 0}" parent="{parent}">'
        cell_xml += f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
        cell_xml += '</mxCell>'
        self.cells.append(cell_xml)
        return cid

    def add_edge(self, source_id, target_id, style="", label="", parent="1"):
        cid = self.next_id()
        lbl_esc = html.escape(label, quote=True)
        edge_xml = f'<mxCell id="{cid}" value="{lbl_esc}" style="{style}" edge="1" source="{source_id}" target="{target_id}" parent="{parent}">'
        edge_xml += '<mxGeometry relative="1" as="geometry" />'
        edge_xml += '</mxCell>'
        self.cells.append(edge_xml)
        return cid

    def build_xml(self):
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml += '<mxfile host="app.diagrams.net" modified="2026-11-13T00:00:00.000Z" agent="Antigravity" version="24.0.0" type="device">\n'
        xml += f'  <diagram id="poster_wd_gpr_sr" name="WD-GPR-SR Academic Poster">\n'
        xml += f'    <mxGraphModel dx="{self.width + 200}" dy="{self.height + 200}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.width}" pageHeight="{self.height}" math="1" shadow="0">\n'
        xml += '      <root>\n'
        xml += '        <mxCell id="0" />\n'
        xml += '        <mxCell id="1" parent="0" />\n'
        for c in self.cells:
            xml += f'        {c}\n'
        xml += '      </root>\n'
        xml += '    </mxGraphModel>\n'
        xml += '  </diagram>\n'
        xml += '</mxfile>'
        return xml

def create_poster():
    b = DrawioPosterBuilder(width=2240, height=1280)
    
    # Outer Canvas Frame
    b.add_cell(
        "", 
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#0f172a;strokeWidth=3;shadow=0;",
        10, 10, 2220, 1260
    )

    # ==========================================
    # 1. HEADER SECTION
    # ==========================================
    # Header background card
    b.add_cell(
        "",
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#0f172a;strokeWidth=2;",
        20, 20, 2200, 120
    )
    
    # Left Emblem: IIT Patna Emblem
    b.add_cell(
        "<b>IIT PATNA</b><br/><font style='font-size: 9px; color: #475569;'>भारतीय प्रौद्योगिकी संस्थान पटना<br/>Est. 2008</font>",
        "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#f1f5f9;strokeColor=#1e3a8a;strokeWidth=2;fontColor=#0f172a;fontSize=11;align=center;",
        35, 30, 100, 100
    )
    
    # Center Title & Author Block
    header_title = (
        "<font style='font-size: 21px; font-weight: 800; color: #0f172a; letter-spacing: -0.2px;'>"
        "Battery State-of-Health Prediction Using Wide-and-Deep Gaussian Process Regression with Symbolic Residual Correction"
        "</font><br/>"
        "<font style='font-size: 14px; font-weight: 600; color: #1e3a8a;'>"
        "Prince Kumar &nbsp;•&nbsp; Roll No: 2401AI38 &nbsp;•&nbsp; Email: 2401ai38@iitp.ac.in &nbsp;•&nbsp; Supervisor: Dr. Jimson Mathew"
        "</font><br/>"
        "<font style='font-size: 12px; color: #334155; font-style: italic;'>"
        "Department of Computer Science &amp; Engineering, Indian Institute of Technology Patna &nbsp;•&nbsp; IMPACT Lab (Embedded AI Initiative)"
        "</font>"
    )
    b.add_cell(
        header_title,
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;",
        145, 25, 1950, 110
    )
    
    # Right Emblem: IMPACT Lab
    b.add_cell(
        "<b>IMPACT</b><br/><font style='font-size: 9px; color: #475569;'>Imaging, Modelling, Perception,<br/>Analytics &amp; Computing<br/>IIT PATNA</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f5f9;strokeColor=#1e3a8a;strokeWidth=2;fontColor=#0f172a;fontSize=11;align=center;",
        2105, 30, 100, 100
    )

    # Dimensions for 3 Columns
    col_top = 150
    col_bottom = 1170
    col_h = col_bottom - col_top # 1020 px
    
    c1_x, c1_w = 20, 520      # Left Column: 520 px
    c2_x, c2_w = 555, 1130    # Center Column: 1130 px (Massive for detailed architecture!)
    c3_x, c3_w = 1700, 520    # Right Column: 520 px

    # Common styles
    card_hdr_style = "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#0f172a;strokeWidth=2;align=left;verticalAlign=top;spacingLeft=12;spacingTop=8;shadow=0;"
    hdr_bar_style = "rounded=0;whiteSpace=wrap;html=1;fillColor=#0f172a;strokeColor=#0f172a;fontColor=#ffffff;fontStyle=1;fontSize=14;align=left;spacingLeft=10;"
    body_txt_style = "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;whiteSpace=wrap;fontSize=11;fontColor=#1e293b;lineSpacing=1.4;"

    # ==========================================
    # 2. LEFT COLUMN (Abstract, Intro, Problem)
    # ==========================================
    # Box 1: Abstract (Y: 150 -> 410, H: 260)
    b.add_cell("", card_hdr_style, c1_x, col_top, c1_w, 260)
    b.add_cell("<b>Abstract</b>", hdr_bar_style, c1_x, col_top, c1_w, 28)
    abs_text = (
        "This work establishes a rigorous physics-guided machine learning framework for <b>Battery State-of-Health (SOH) prognostics</b> across diverse cell chemistries (LiCoO₂ and LiNiMnCoO₂). "
        "Evaluated on <b>4,657 laboratory cycles</b> from NASA Ames and CALCE CS2 repositories under strict <b>Leave-One-Cell-Out Cross-Validation (LOCO-CV)</b>, standard machine learning methods suffer severe performance degradation or overconfidence. "
        "To overcome these bottlenecks, we propose <b>Wide-and-Deep Sparse Variational Gaussian Process Regression with Uncertainty-Weighted Symbolic Residual Correction (WD-GPR-SR)</b>.<br/><br/>"
        "The architecture integrates a dual-branch compound kernel capturing both linear aging trajectories and deep latent degradation manifolds, paired with parsimonious genetic programming trained on <i>honest inner out-of-fold residuals</i>. "
        "The model achieves an outstanding <b>RMSE of 1.64%</b> on CALCE with <b>94.5% 95-PICP coverage</b> and executes in <b>0.15 ms</b> on an NVIDIA Jetson Orin Nano with zero PyTorch runtime dependencies."
    )
    b.add_cell(abs_text, body_txt_style, c1_x + 12, col_top + 36, c1_w - 24, 215)

    # Box 2: Introduction (Y: 420 -> 870, H: 450)
    intro_top = col_top + 270
    b.add_cell("", card_hdr_style, c1_x, intro_top, c1_w, 450)
    b.add_cell("<b>Introduction &amp; Research Gaps</b>", hdr_bar_style, c1_x, intro_top, c1_w, 28)
    intro_text = (
        "<b>Mission-Critical EV &amp; Grid Energy Storage:</b> Accurate SOH prognostics is vital for battery safety, remaining useful life (RUL) estimation, and thermal runaway prevention.<br/><br/>"
        "<b>Critical Vulnerabilities in Existing Paradigms:</b><br/>"
        "❑ <b>Cross-Cycle Data Leakage:</b> Prevalent battery ML literature employs random train/test cycle splits. Because temporal autocorrelations leak across adjacent cycles, models report artificially high R² &gt; 0.99 that completely collapse on unseen battery cells. Genuine cross-cell LOCO-CV is non-negotiable.<br/><br/>"
        "❑ <b>Black-Box Deep Neural Overconfidence:</b> Deep MLPs and LSTMs lack epistemic uncertainty quantification. In safety-critical automotive BMS, deterministic models exhibit fatal overconfidence during sudden non-linear knee-point fade.<br/><br/>"
        "❑ <b>O(N³) Scalability Bottleneck of Exact GPR:</b> Exact Gaussian Processes require full N×N kernel matrix inversion, requiring gigabytes of RAM and rendering continuous real-time execution impossible on microcontrollers.<br/><br/>"
        "❑ <b>Near-Zero-Residual Trap in Naive Hybrid Models:</b> Standard symbolic regression trained on in-sample GP residuals collapses because the flexible GP interpolates training points (residuals ≈ 0), causing genetic search to fit noise or select constant zero."
    )
    b.add_cell(intro_text, body_txt_style, c1_x + 12, intro_top + 36, c1_w - 24, 405)

    # Box 3: Problem Statement (Y: 880 -> 1170, H: 290)
    prob_top = intro_top + 460
    b.add_cell("", card_hdr_style, c1_x, prob_top, c1_w, 290)
    b.add_cell("<b>Problem Statement &amp; Objectives</b>", hdr_bar_style, c1_x, prob_top, c1_w, 28)
    prob_text = (
        "<b>Formal Mathematical Formulation:</b><br/>"
        "Given pre-discharge health indicators $\\mathbf{x} \\in \\mathbb{R}^5$ per cycle, predict ground-truth capacity percentage $y \\in [0, 100]$ under three strict engineering requirements:<br/><br/>"
        "1. <b>Zero-Leakage Generalization:</b> Train exclusively on cells $\\mathcal{C}_{\\text{train}}$ and evaluate on held-out cell $\\mathcal{C}_{\\text{test}}$ without target cell calibration.<br/>"
        "2. <b>Reliable Uncertainty Bounds:</b> Produce analytical epistemic variance $\\sigma^2(\\mathbf{x})$ maintaining &ge; 90% empirical coverage (95% PICP) with minimal interval width (MPIW).<br/>"
        "3. <b>Interpretable Edge Deployment:</b> Discover a parsimonious closed-form residual equation $g(\\mathbf{x})$, deploying the entire inference pipeline in standalone C + ONNX Runtime on NVIDIA Jetson Orin Nano with &lt; 1 ms latency."
    )
    b.add_cell(prob_text, body_txt_style, c1_x + 12, prob_top + 36, c1_w - 24, 245)

    # ==========================================
    # 3. CENTER COLUMN (Contributions, Architecture, Algorithm)
    # ==========================================
    # Box 1: Key Contributions (Y: 150 -> 280, H: 130)
    b.add_cell("", card_hdr_style, c2_x, col_top, c2_w, 130)
    b.add_cell("<b>Key Contributions</b>", hdr_bar_style, c2_x, col_top, c2_w, 28)
    contrib_text = (
        "❑ <b>Wide-and-Deep Sparse Variational GPR:</b> Formulated a unified compound kernel combining linear trend, ARD-RBF smooth dynamics, and deep Matérn-5/2 manifold projection with M=64 inducing points, slashing computational complexity from O(N³) to O(M²N).<br/>"
        "❑ <b>Honest Inner LOCO-CV &amp; Uncertainty-Weighted SR:</b> Developed an out-of-fold residual extraction protocol with inverse-variance sample weights $w_i = \\text{clip}(1/(\\sigma_i^2 + \\epsilon))$, defeating the near-zero residual trap without test data leakage.<br/>"
        "❑ <b>Discovered 51-Node Transparent Closed-Form Equation:</b> Mined an interpretable mathematical formula correcting non-linear constant-voltage charge saturation and incremental capacity voltage shifts.<br/>"
        "❑ <b>Zero-PyTorch Edge Engine:</b> Implemented a standalone ANSI C + ONNX Runtime inference engine delivering 0.15 ms latency and 6,395 inferences/s on Jetson Orin Nano."
    )
    b.add_cell(contrib_text, body_txt_style, c2_x + 12, col_top + 34, c2_w - 24, 90)

    # Box 2: Methodology & Detailed Central Architecture (Y: 290 -> 875, H: 585)
    arch_top = col_top + 140
    arch_h = 585
    b.add_cell("", card_hdr_style, c2_x, arch_top, c2_w, arch_h)
    b.add_cell("<b>Methodology &amp; Detailed System Architecture</b>", hdr_bar_style, c2_x, arch_top, c2_w, 28)

    # -------------------------------------------------------------------------
    # DETAILED CENTRAL ARCHITECTURE DIAGRAM (Figure 1)
    # -------------------------------------------------------------------------
    fig_x = c2_x + 15
    fig_y = arch_top + 35
    fig_w = c2_w - 30
    fig_h = arch_h - 60

    # Sub-card: Figure 1 Outer Container
    b.add_cell(
        "",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#cbd5e1;strokeWidth=1.5;dashed=0;",
        fig_x, fig_y, fig_w, fig_h
    )

    # Stage 1: Feature Extraction Card (Left)
    s1_w = 175
    s1_h = 440
    s1_x = fig_x + 12
    s1_y = fig_y + 15
    b.add_cell("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#0284c7;strokeWidth=1.5;", s1_x, s1_y, s1_w, s1_h)
    b.add_cell("<b>1. Pre-Discharge Features</b><br/><font style='font-size:9px; color:#0369a1;'>Raw Battery Cycling</font>", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e0f2fe;strokeColor=#0284c7;fontColor=#0369a1;fontSize=10;align=center;", s1_x+5, s1_y+6, s1_w-10, 32)
    
    feats = [
        ("x₀: Cycle Count", "Aging progression index"),
        ("x₁: CC Charge Time", "Bulk lithium intercalation duration"),
        ("x₂: CV Charge Time", "Diffusion saturation relaxation time"),
        ("x₃: IC Peak Height", "dQ/dV active phase transition peak"),
        ("x₄: IC Peak Voltage", "Overpotential degradation shift")
    ]
    feat_y = s1_y + 46
    feat_node_ids = []
    for fname, fdesc in feats:
        fid = b.add_cell(
            f"<b>{fname}</b><br/><font style='font-size:8px; color:#475569;'>{fdesc}</font>",
            "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#94a3b8;fontSize=9;align=center;shadow=0;",
            s1_x + 8, feat_y, s1_w - 16, 42
        )
        feat_node_ids.append(fid)
        feat_y += 50

    b.add_cell(
        "<b>Feature Vector</b><br/>$$\\mathbf{x} \\in \\mathbb{R}^5$$<br/><font style='font-size:9px; color:#64748b;'>Standardized (μ=0, σ=1)</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f5f9;strokeColor=#64748b;fontSize=10;align=center;",
        s1_x + 8, feat_y + 10, s1_w - 16, 55
    )

    # Stage 2: Wide-and-Deep Compound Kernel (Center-Left)
    s2_w = 265
    s2_h = 440
    s2_x = s1_x + s1_w + 35
    s2_y = s1_y
    b.add_cell("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#4f46e5;strokeWidth=1.5;", s2_x, s2_y, s2_w, s2_h)
    b.add_cell("<b>2. Wide &amp; Deep Kernel Engine</b><br/><font style='font-size:9px; color:#3730a3;'>Compound Physics Representation</font>", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ede9fe;strokeColor=#4f46e5;fontColor=#3730a3;fontSize=10;align=center;", s2_x+5, s2_y+6, s2_w-10, 32)

    # Wide branch box
    wide_box = b.add_cell(
        "<b>WIDE BRANCH (Global Trend &amp; ARD)</b><br/>"
        "<font style='font-size:9px; color:#1e293b;'>"
        "• Linear Kernel: $$k_{\\text{lin}}(\\mathbf{x}, \\mathbf{x}') = \\mathbf{x}^\\top \\mathbf{x}'$$<br/>"
        "• ARD-RBF Kernel:<br/>"
        "$$k_{\\text{ARD}} = \\exp\\left(-\\frac{1}{2}\\sum_{d=1}^5 \\frac{(x_d - x'_d)^2}{l_d^2}\\right)$$<br/>"
        "<i>Preserves baseline linear capacity decline and length-scale physics.</i>"
        "</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#3b82f6;strokeWidth=1.2;fontSize=9.5;align=left;spacingLeft=6;spacingRight=6;",
        s2_x + 10, s2_y + 46, s2_w - 20, 115
    )

    # Deep branch box
    deep_box = b.add_cell(
        "<b>DEEP BRANCH (Latent Manifold)</b><br/>"
        "<font style='font-size:9px; color:#1e293b;'>"
        "• MLP: $$\\mathbf{x} \\xrightarrow{\\phi_\\theta} [32 \\to 16] \\to \\mathbf{z} \\in \\mathbb{R}^8$$<br/>"
        "• Matérn-5/2 Kernel on Manifold:<br/>"
        "$$k_{\\text{Mat}}(d) = \\left(1 + \\sqrt{5}d + \\frac{5}{3}d^2\\right)e^{-\\sqrt{5}d}$$<br/>"
        "<i>Discovers non-linear electrochemical phase transitions.</i>"
        "</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#8b5cf6;strokeWidth=1.2;fontSize=9.5;align=left;spacingLeft=6;spacingRight=6;",
        s2_x + 10, s2_y + 170, s2_w - 20, 120
    )

    # Compound summation
    kernel_sum = b.add_cell(
        "<b>Compound Wide-and-Deep Kernel</b><br/>"
        "$$k_{\\text{WD}}(\\mathbf{x}, \\mathbf{x}') = \\sigma_w^2 \\left[k_{\\text{lin}} + k_{\\text{ARD}}\\right] + \\sigma_d^2 k_{\\text{Mat}}(\\mathbf{z}, \\mathbf{z}')$$",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f5f9;strokeColor=#475569;strokeWidth=1.2;fontSize=9.5;align=center;",
        s2_x + 10, s2_y + 300, s2_w - 20, 58
    )

    # Sparse Variational GP block
    svgp_box = b.add_cell(
        "<b>Sparse Variational GP ($M=64$ Inducing Points)</b><br/>"
        "<font style='font-size:8.5px; color:#334155;'>"
        "$\\mathcal{L}_{\\text{ELBO}} = \\sum_{i=1}^N \\mathbb{E}[\\log p(y_i|f_i)] - \\text{KL}(q(\\mathbf{u}) \\parallel p(\\mathbf{u}))$<br/>"
        "Complexity: $\\mathcal{O}(M^2 N)$ instead of $\\mathcal{O}(N^3)$"
        "</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#fef3c7;strokeColor=#d97706;strokeWidth=1.2;fontSize=9.5;align=center;",
        s2_x + 10, s2_y + 368, s2_w - 20, 60
    )

    # Stage 3: Honest Inner LOCO-CV & Uncertainty Weighting (Center-Right)
    s3_w = 275
    s3_h = 440
    s3_x = s2_x + s2_w + 35
    s3_y = s1_y
    b.add_cell("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#059669;strokeWidth=1.5;", s3_x, s3_y, s3_w, s3_h)
    b.add_cell("<b>3. Uncertainty-Weighted SR Engine</b><br/><font style='font-size:9px; color:#065f46;'>Zero-Leakage Residual Correction</font>", "rounded=1;whiteSpace=wrap;html=1;fillColor=#d1fae5;strokeColor=#059669;fontColor=#065f46;fontSize=10;align=center;", s3_x+5, s3_y+6, s3_w-10, 32)

    # Trap & Solution
    trap_box = b.add_cell(
        "<b>The Near-Zero-Residual Trap Solved:</b><br/>"
        "<font style='font-size:9px; color:#1e293b;'>"
        "• In-sample residuals $r_i^{\\text{train}} \\approx 0$ collapse SR to 0.<br/>"
        "• <b>Inner 3-Cell LOCO-CV:</b> Excludes cell $k$ during residual generation:<br/>"
        "$$r_i^{\\text{OOF}} = y_i - \\mu_{\\text{GP}}^{(-k)}(\\mathbf{x}_i), \\quad \\sigma_i^{\\text{OOF}} = \\sigma_{\\text{GP}}^{(-k)}(\\mathbf{x}_i)$$<br/>"
        "<i>Guarantees authentic out-of-fold generalization error.</i>"
        "</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#10b981;strokeWidth=1.2;fontSize=9.5;align=left;spacingLeft=6;spacingRight=6;",
        s3_x + 10, s3_y + 46, s3_w - 20, 115
    )

    # Uncertainty Sample Weighting
    weight_box = b.add_cell(
        "<b>Uncertainty Inverse-Variance Weighting:</b><br/>"
        "<font style='font-size:9px; color:#1e293b;'>"
        "$$w_i = \\text{clip}\\left(\\frac{1}{(\\sigma_i^{\\text{OOF}})^2 + 10^{-6}},\\; q_{1\\%},\\; q_{99\\%}\\right)$$<br/>"
        "• High certainty, non-zero error: $w_i \\gg 1$ &rarr; forces fit.<br/>"
        "• High uncertainty (knee fade): $w_i \\ll 1$ &rarr; ignores noise."
        "</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#059669;strokeWidth=1.2;fontSize=9.5;align=left;spacingLeft=6;spacingRight=6;",
        s3_x + 10, s3_y + 170, s3_w - 20, 95
    )

    # Genetic Programming Search
    gp_sr_box = b.add_cell(
        "<b>Parsimonious Genetic Programming:</b><br/>"
        "<font style='font-size:9px; color:#1e293b;'>"
        "$$\\min_{g} \\frac{1}{N}\\sum_{i=1}^N w_i \\left(r_i^{\\text{OOF}} - g(\\mathbf{x}_i)\\right)^2 + \\lambda \\cdot \\text{Complexity}(g)$$<br/>"
        "Protected operators: \{+, -, \\times, \\div, \\sqrt{\\cdot}, \\ln|\\cdot|\}$, $\\lambda = 0.001$<br/>"
        "<b>Result:</b> Discovers compact formula $g^*(\mathbf{x})$ (51 nodes)."
        "</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#059669;strokeWidth=1.2;fontSize=9.5;align=left;spacingLeft=6;spacingRight=6;",
        s3_x + 10, s3_y + 275, s3_w - 20, 95
    )

    # Shrunk / Selection Gate
    gate_box = b.add_cell(
        "<b>Nested Cross-Cell SR Selection Gate:</b><br/>"
        "<font style='font-size:8.5px; color:#065f46;'>"
        "Selects $g^*(\mathbf{x})$ ONLY if inner CV RMSE improves over base GP;<br/>"
        "otherwise gracefully falls back to $g(\\mathbf{x}) \\equiv 0$ (prevents overfit)."
        "</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ecfdf5;strokeColor=#059669;strokeWidth=1;fontSize=9;align=center;",
        s3_x + 10, s3_y + 380, s3_w - 20, 48
    )

    # Stage 4: Final Inference & Calibrated Output (Right)
    s4_w = 250
    s4_h = 440
    s4_x = s3_x + s3_w + 35
    s4_y = s1_y
    b.add_cell("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#e11d48;strokeWidth=1.5;", s4_x, s4_y, s4_w, s4_h)
    b.add_cell("<b>4. Hybrid Prediction &amp; Edge</b><br/><font style='font-size:9px; color:#9f1239;'>Zero-PyTorch Runtime</font>", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe4e6;strokeColor=#e11d48;fontColor=#9f1239;fontSize=10;align=center;", s4_x+5, s4_y+6, s4_w-10, 32)

    # Final closed-form prediction equation
    out_pred = b.add_cell(
        "<b>Hybrid Closed-Form Prediction:</b><br/>"
        "<font style='font-size:11px; color:#0f172a;'>$$\\hat{y}(\\mathbf{x}) = \\mu_{\\text{GP}}(\\mathbf{x}) + g^*(\mathbf{x})$$</font><br/>"
        "<font style='font-size:8.5px; color:#475569;'>GP smooth global prior + Symbolic explicit correction</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#f43f5e;strokeWidth=1.2;fontSize=9.5;align=center;",
        s4_x + 10, s4_y + 46, s4_w - 20, 75
    )

    # Calibrated 95% Confidence Interval
    out_ci = b.add_cell(
        "<b>Calibrated 95% Confidence Interval:</b><br/>"
        "<font style='font-size:10px; color:#0f172a;'>$$\\mathcal{I}_{95\\%} = \\left[ \\hat{y}(\\mathbf{x}) \\pm 1.96 \\cdot s \\cdot \\sigma_{\\text{GP}}(\\mathbf{x}) \\right]$$</font><br/>"
        "<font style='font-size:8.5px; color:#475569;'>Where $s$ is calibrated scaling factor from OOF validation.</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#f43f5e;strokeWidth=1.2;fontSize=9.5;align=center;",
        s4_x + 10, s4_y + 130, s4_w - 20, 80
    )

    # Standalone Edge Engine Card
    edge_box = b.add_cell(
        "<b>Standalone Embedded Edge Engine</b><br/>"
        "<font style='font-size:8.5px; color:#1e293b;'>"
        "• <b>Zero PyTorch Dependency:</b> Pure C + ONNX<br/>"
        "• <code>scaler_params.npz</code>: Normalization<br/>"
        "• <code>phi_theta.onnx</code>: Deep manifold latent<br/>"
        "• <code>svgp_arrays.npz</code>: Inducing $\\mathbf{Z}, \\alpha, \\mathbf{A}$<br/>"
        "• <code>libequation.so</code>: Compiled C equation<br/>"
        "-------------------------------------<br/>"
        "<b>Target: NVIDIA Jetson Orin Nano (8GB)</b><br/>"
        "&bull; Median Latency: <b>0.15 ms</b> (153 &mu;s)<br/>"
        "&bull; Throughput: <b>6,395 inferences / sec</b><br/>"
        "&bull; RAM Delta: <b>0.49 MB</b> (Ultra-lean)"
        "</font>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#fef2f2;strokeColor=#e11d48;strokeWidth=1.2;fontSize=9;align=left;spacingLeft=8;",
        s4_x + 10, s4_y + 220, s4_w - 20, 208
    )

    # Flow arrows inside Figure 1
    edge_style = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#0f172a;strokeWidth=1.8;endArrow=block;endFill=1;"
    dashed_edge = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#059669;strokeWidth=1.5;dashed=1;endArrow=block;endFill=1;"
    
    b.add_edge(s1_x + s1_w, s2_x, edge_style, "")
    b.add_edge(s2_x + s2_w, s3_x, edge_style, "μ_GP, σ_GP")
    b.add_edge(s3_x + s3_w, s4_x, edge_style, "g*(x)")
    b.add_edge(s2_x + s2_w, s4_x + 30, "edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#4f46e5;strokeWidth=1.5;endArrow=block;", "μ_GP(x)")

    # Caption for Figure 1
    b.add_cell(
        "<b>Figure 1: End-to-End System Architecture of Wide-and-Deep Gaussian Process Regression with Uncertainty-Weighted Symbolic Correction (WD-GPR-SR)</b>",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11;fontColor=#0f172a;fontStyle=1;",
        fig_x, fig_y + fig_h - 22, fig_w, 20
    )

    # Box 3: Algorithm 1 (Y: 885 -> 1170, H: 285)
    algo_top = arch_top + arch_h + 10
    algo_h = 275
    b.add_cell("", card_hdr_style, c2_x, algo_top, c2_w, algo_h)
    b.add_cell("<b>Algorithm 1: Zero-Leakage WD-GPR-SR Prognostic Pipeline</b>", hdr_bar_style, c2_x, algo_top, c2_w, 28)
    
    algo_text = (
        "<b>Require:</b> Cell dataset $\\mathcal{D} = \\{(\\mathbf{x}_i, y_i)\\}_{i=1}^N$ with cells $\\mathcal{C} = \\{c_1, c_2, c_3, c_4\\}$; inducing points $M=64$; parsimony $\\lambda=0.001$; test cell $c_{\\text{test}}$.<br/>"
        "<b>Ensure:</b> Calibrated hybrid predictor $\\hat{y}(\\mathbf{x})$ and 95% confidence interval $\\mathcal{I}_{95\\%}(\\mathbf{x})$ on unseen held-out cell $c_{\\text{test}}$.<br/>"
        "&nbsp;1: <b>Partition Cells:</b> Set training pool $\\mathcal{C}_{\\text{train}} = \\mathcal{C} \\setminus \\{c_{\\text{test}}\\}$.<br/>"
        "&nbsp;2: <b>Inner Leave-One-Cell-Out Cross-Validation:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>for</b> each training cell $k \\in \\mathcal{C}_{\\text{train}}$ <b>do</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Train inner WD-SVGP model $\\mathcal{M}_{\\text{inner}}$ on $\\mathcal{C}_{\\text{train}} \\setminus \\{k\\}$ with compound kernel $k_{\\text{WD}}$.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Evaluate on held-out training cell $k$: predict out-of-fold mean $\\mu_i^{\\text{OOF}}$ and epistemic variance $(\\sigma_i^{\\text{OOF}})^2$.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Record honest out-of-fold residuals: $r_i^{\\text{OOF}} = y_i - \\mu_i^{\\text{OOF}}$, $\\forall i \\in k$.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>end for</b><br/>"
        "&nbsp;3: <b>Uncertainty Weight Computation:</b> Calculate sample weights $w_i = \\text{clip}\\left(1 / [(\\sigma_i^{\\text{OOF}})^2 + 10^{-6}], q_{1\\%}, q_{99\\%}\\right)$ normalized such that $\\sum w_i = N_{\\text{train}}$.<br/>"
        "&nbsp;4: <b>Parsimonious Symbolic Evolution:</b> Run genetic tournament search minimizing $\\mathcal{L}_{\\text{SR}}(g, \\mathbf{w})$; select candidate $g^*$ via nested cross-validation validation.<br/>"
        "&nbsp;5: <b>Train Primary WD-SVGP Model:</b> Fit full WD-SVGP on all cells in $\\mathcal{C}_{\\text{train}}$ using Adam (25 epochs, lr=0.015) to obtain $(\\mu_{\\text{GP}}, \\sigma_{\\text{GP}}^2)$.<br/>"
        "&nbsp;6: <b>Recalibrate Variance:</b> Calculate variance scaling factor $s^2 = \\frac{1}{N} \\sum (r_i^{\\text{OOF}})^2 / (\\sigma_i^{\\text{OOF}})^2$.<br/>"
        "&nbsp;7: <b>Deploy &amp; Predict on $c_{\\text{test}}$:</b> $\\hat{y}(\\mathbf{x}) = \\mu_{\\text{GP}}(\\mathbf{x}) + g^*(\mathbf{x})$, \\; with 95% CI: $[\\hat{y} \\pm 1.96 \\cdot s \\cdot \\sigma_{\\text{GP}}(\\mathbf{x})]$."
    )
    b.add_cell(algo_text, body_txt_style, c2_x + 12, algo_top + 34, c2_w - 24, 230)

    # ==========================================
    # 4. RIGHT COLUMN (Results, Equation, Conclusion)
    # ==========================================
    # Box 1: Results and Discussion (Y: 150 -> 640, H: 490)
    b.add_cell("", card_hdr_style, c3_x, col_top, c3_w, 490)
    b.add_cell("<b>Result and Discussion</b>", hdr_bar_style, c3_x, col_top, c3_w, 28)

    table_html = (
        "<b>Table 1: Cross-Domain Benchmark vs. Competing Models</b><br/>"
        "<font style='font-size:8.5px; color:#475569;'>Verified Leave-One-Cell-Out Cross-Validation (4,657 Total Cycles)</font><br/>"
        "<table border='1' cellspacing='0' cellpadding='3' style='border-collapse:collapse; width:100%; font-size:9.5px; text-align:center; border: 1px solid #94a3b8; margin-top:4px;'>"
        "<tr style='background-color:#0f172a; color:#ffffff; font-weight:bold;'>"
        "  <td>Model</td><td>MAE(%)</td><td>RMSE(%)</td><td>R²</td><td>95%PICP</td><td>Lat.</td>"
        "</tr>"
        "<tr style='background-color:#ffffff;'>"
        "  <td align='left'>Linear Ridge</td><td>4.82</td><td>6.42</td><td>0.682</td><td>58.2%</td><td>0.02ms</td>"
        "</tr>"
        "<tr style='background-color:#f8fafc;'>"
        "  <td align='left'>Decision Tree</td><td>3.95</td><td>5.73</td><td>0.741</td><td>&mdash;</td><td>0.05ms</td>"
        "</tr>"
        "<tr style='background-color:#ffffff;'>"
        "  <td align='left'>Random Forest</td><td>3.12</td><td>4.95</td><td>0.812</td><td>&mdash;</td><td>1.84ms</td>"
        "</tr>"
        "<tr style='background-color:#f8fafc;'>"
        "  <td align='left'>Deep MLP</td><td>2.85</td><td>5.21</td><td>0.791</td><td>&mdash;</td><td>0.45ms</td>"
        "</tr>"
        "<tr style='background-color:#ffffff;'>"
        "  <td align='left'>Vanilla SVGP</td><td>2.14</td><td>3.51</td><td>0.865</td><td>71.4%</td><td>0.12ms</td>"
        "</tr>"
        "<tr style='background-color:#ecfdf5; font-weight:bold; color:#065f46; border:2px solid #059669;'>"
        "  <td align='left'>Proposed (WD-GPR-SR)</td><td><b>0.97</b></td><td><b>1.64</b></td><td><b>0.978</b></td><td><b>94.5%</b></td><td><b>0.15ms</b></td>"
        "</tr>"
        "</table>"
    )
    b.add_cell(table_html, body_txt_style, c3_x + 12, col_top + 34, c3_w - 24, 160)

    res_notes = (
        "<b>Key Experimental Observations:</b><br/>"
        "❑ <b>Massive Accuracy Gains:</b> Proposed WD-GPR-SR delivers a <b>66.8% reduction in MAE</b> over Linear Regression and <b>53.3% lower RMSE</b> than Deep MLP, proving that combining wide linear trends with deep manifold representations captures complex degradation.<br/><br/>"
        "❑ <b>Calibrated Epistemic Uncertainty:</b> Re-calibrated 95% Prediction Interval Coverage Probability (PICP) reaches <b>94.5%</b> on CALCE (ideal target: 95.0%) with tight Mean Prediction Interval Width (MPIW) of 6.50%, providing guaranteed safety bounds for BMS controllers.<br/><br/>"
        "❑ <b>Component Ablation Progression:</b><br/>"
        "&nbsp;&bull; <i>Config A (Vanilla GPR):</i> RMSE 2.95% (CALCE) | 3.51% (NASA)<br/>"
        "&nbsp;&bull; <i>Config B (Deep Kernel):</i> RMSE 2.81% (captures knee fade)<br/>"
        "&nbsp;&bull; <i>Config C (Wide+Deep SVGP):</i> RMSE 2.53%, MAE 1.25%<br/>"
        "&nbsp;&bull; <i>Config D (WD-SVGP + UW-SR):</i> <b>RMSE 1.64%, MAE 0.97%</b>"
    )
    b.add_cell(res_notes, body_txt_style, c3_x + 12, col_top + 205, c3_w - 24, 275)

    # Box 2: Discovered Equation & Interpretation (Y: 650 -> 935, H: 285)
    eq_top = col_top + 500
    b.add_cell("", card_hdr_style, c3_x, eq_top, c3_w, 285)
    b.add_cell("<b>Discovered Closed-Form Residual Equation</b>", hdr_bar_style, c3_x, eq_top, c3_w, 28)

    eq_body = (
        "<b>Symbolic Genetic Programming Output $g^*(\mathbf{x})$:</b><br/>"
        "<font style='font-size:9.5px; color:#0f172a; font-family:Courier New, monospace;'>"
        "$$g^*(\mathbf{x}) = -x_{\\text{CCCT}} \\cdot x_{\\text{IC\\_h}} + x_{\\text{CCCT}} + 3x_{\\text{cycle}}$$<br/>"
        "$$\\quad\\quad - 5x_{\\text{IC\\_v}} - \\ln(|x_{\\text{CVCT}}| + 10^{-6}) - \\sqrt{|x_{\\text{IC\\_v}} + 0.492|}$$"
        "</font><br/>"
        "<font style='font-size:9px; color:#475569;'>"
        "• <b>Structural Complexity:</b> 51 syntax tree nodes (33 mathematical operators).<br/>"
        "• <b>Zero-PyTorch C Code:</b> Compiled to <code>libequation.so</code> executed via <code>ctypes</code>.<br/>"
        "• <b>Physical Attribution:</b><br/>"
        "&nbsp;&nbsp;- $x_{\\text{CCCT}} \\cdot x_{\\text{IC\\_h}}$ couples constant-current charging time to phase transition peak height.<br/>"
        "&nbsp;&nbsp;- $-\\ln(|x_{\\text{CVCT}}|)$ models asymptotic charge saturation during constant-voltage plateau.<br/>"
        "&nbsp;&nbsp;- $-5x_{\\text{IC\\_v}}$ captures accelerated voltage overpotential shift near end-of-life."
        "</font>"
    )
    b.add_cell(eq_body, body_txt_style, c3_x + 12, eq_top + 34, c3_w - 24, 240)

    # Box 3: Conclusion (Y: 945 -> 1170, H: 225)
    conc_top = eq_top + 295
    b.add_cell("", card_hdr_style, c3_x, conc_top, c3_w, 225)
    b.add_cell("<b>Conclusion &amp; Future Outlook</b>", hdr_bar_style, c3_x, conc_top, c3_w, 28)
    conc_text = (
        "❑ <b>Established Zero-Leakage Benchmark:</b> Demonstrated that Leave-One-Cell-Out Cross-Validation is essential to eliminate false high-accuracy illusions in battery prognostics.<br/><br/>"
        "❑ <b>Defeated Near-Zero Residual Trap:</b> Honest inner LOCO uncertainty-weighted symbolic regression mines genuine cross-cell generalization corrections without test leakage.<br/><br/>"
        "❑ <b>Sub-Millisecond Edge Feasibility:</b> The complete zero-PyTorch pipeline achieves <b>0.15 ms latency</b>, enabling real-time continuous cycle-by-cycle SOH estimation in commercial Electric Vehicle BMS units."
    )
    b.add_cell(conc_text, body_txt_style, c3_x + 12, conc_top + 34, c3_w - 24, 185)

    # ==========================================
    # 5. FOOTER / REFERENCES (Y: 1180 -> 1260, H: 80)
    # ==========================================
    foot_y = 1180
    b.add_cell("", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#0f172a;strokeWidth=2;", 20, foot_y, 2200, 80)
    b.add_cell("<b>References:</b>", "text;html=1;strokeColor=none;fillColor=none;fontColor=#0f172a;fontSize=10;fontStyle=1;align=left;", 28, foot_y + 4, 150, 18)
    
    refs_text = (
        "[1] Saha, B., &amp; Goebel, K. (2007). <i>Battery Data Set</i>. NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.<br/>"
        "[2] He, W., Williard, N., Osterman, M., &amp; Pecht, M. (2011). Prognostics of lithium-ion batteries based on Dempster-Shafer theory and Bayesian Monte Carlo. <i>IEEE Trans. Ind. Electron.</i>, 58(9), 4398–4405.<br/>"
        "[3] Hensman, J., Fusi, N., &amp; Lawrence, N. D. (2013). Gaussian Processes for Big Data. <i>Proc. Uncertainty in Artificial Intelligence (UAI)</i>, 282–290.<br/>"
        "[4] Wilson, A. G., Hu, Z., Salakhutdinov, R., &amp; Xing, E. P. (2016). Deep Kernel Learning. <i>Proc. AISTATS</i>, 370–378. &nbsp;|&nbsp; [5] Koza, J. R. (1992). <i>Genetic Programming</i>. MIT Press."
    )
    b.add_cell(refs_text, "text;html=1;strokeColor=none;fillColor=none;fontColor=#334155;fontSize=9;align=left;lineSpacing=1.2;", 28, foot_y + 22, 2180, 52)

    return b.build_xml()

def compress_drawio_xml(xml_content):
    quoted = urllib.parse.quote(xml_content, safe="~()*!'")
    compressor = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15, memLevel=8, strategy=0)
    compressed = compressor.compress(quoted.encode('utf-8')) + compressor.flush()
    b64 = base64.b64encode(compressed).decode('utf-8')
    return "https://app.diagrams.net/#R" + urllib.parse.quote(b64)

def main():
    xml_str = create_poster()
    out_path = "2401AI38_MLProject_Poster.drawio"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"Generated draw.io poster successfully: {out_path} ({len(xml_str)} bytes)")
    
    url = compress_drawio_xml(xml_str)
    url_path = "2401AI38_MLProject_Poster_url.txt"
    with open(url_path, "w", encoding="utf-8") as f:
        f.write(url)
    print(f"Generated instant 1-click draw.io URL saved to: {url_path} ({len(url)} chars)")

if __name__ == "__main__":
    main()
