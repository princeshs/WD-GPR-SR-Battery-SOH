#!/usr/bin/env python3
"""
Render crisp, publication-quality LaTeX equation figures for WD-GPR-SR.
Saved to results/figures/equations/*.png

Fixes:
- Replaces invisible transparent backgrounds with solid white cards (#ffffff)
- Adds a subtle boundary (#cbd5e1) so cards stand out in dark-mode image viewers
- Formats multi-part expressions across balanced lines for readable aspect ratios
- Ensures high-DPI crisp Computer Modern LaTeX font
"""

import os
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

plt.rcParams['mathtext.fontset'] = 'cm'

EQUATIONS = {
    'eq0_dkl.png': (
        r'$k_{\mathrm{DKL}}(\mathbf{x}, \mathbf{x}^\prime) = k_0\left(\phi_\theta(\mathbf{x}), \phi_\theta(\mathbf{x}^\prime)\right)$',
        16, 170
    ),
    'eq0_soh.png': (
        r'$\mathrm{SOH}_k = \frac{C_k}{C_{\mathrm{ref}}} \times 100\%, \qquad C_{\mathrm{ref}} = \mathrm{median}\left(C_1, C_2, C_3\right)$',
        16, 170
    ),
    'eq1_overall.png': (
        r'$\hat{y}(\mathbf{x}) = \mu_{\mathrm{GP}}(\mathbf{x}) + g^*(\mathbf{x}), \qquad \mathbf{x} \in \mathbb{R}^D$',
        17, 170
    ),
    'eq2_kernel.png': (
        r'$k_{\mathrm{WD}}(\mathbf{x}, \mathbf{x}^\prime) = \sigma_w^2 \left[ k_{\mathrm{lin}}(\mathbf{x}, \mathbf{x}^\prime) + k_{\mathrm{RBF}\text{-}\mathrm{ARD}}(\mathbf{x}, \mathbf{x}^\prime) \right] + \sigma_d^2 \, k_{\mathrm{deep}}(\mathbf{x}, \mathbf{x}^\prime)$'
        '\n'
        r'$\text{where} \quad k_{\mathrm{deep}}(\mathbf{x}, \mathbf{x}^\prime) = k_{\mathrm{Mat\acute{e}rn}\text{-}5/2}\left(\phi_\theta(\mathbf{x}), \phi_\theta(\mathbf{x}^\prime)\right), \quad k_{\mathrm{lin}}(\mathbf{x}, \mathbf{x}^\prime) = \mathbf{x}^\top \mathbf{x}^\prime$'
        '\n'
        r'$\text{and} \quad k_{\mathrm{RBF}\text{-}\mathrm{ARD}}(\mathbf{x}, \mathbf{x}^\prime) = \exp\left(-\frac{1}{2}\sum_{d=1}^D \frac{(x_d - x_d^\prime)^2}{l_d^2}\right)$',
        13, 260
    ),
    'eq3_subkernels.png': (
        r'$k_{\mathrm{wide}}(\mathbf{x}, \mathbf{x}^\prime) = k_{\mathrm{linear}}(\mathbf{x}, \mathbf{x}^\prime) + k_{\mathrm{ARD}\text{-}\mathrm{RBF}}(\mathbf{x}, \mathbf{x}^\prime)$'
        '\n'
        r'$k_{\mathrm{deep}}(\mathbf{x}, \mathbf{x}^\prime) = k_{\mathrm{Mat\acute{e}rn}\text{-}5/2}\left(\phi_\theta(\mathbf{x}), \phi_\theta(\mathbf{x}^\prime)\right)$',
        14, 210
    ),
    'eq3_elbo.png': (
        r'$\mathcal{L}_{\mathrm{ELBO}}(\theta, \mathbf{Z}, \mathbf{m}, \mathbf{S}) = \sum_{i=1}^N \mathbb{E}_{q(f_i)}\left[\log p(y_i \mid f_i)\right] - \mathrm{KL}\left(q(\mathbf{u}) \,\Vert\, p(\mathbf{u})\right)$',
        15, 180
    ),
    'eq4_elbo.png': (
        r'$\mathcal{L}_{\mathrm{ELBO}}(\theta, \mathbf{Z}, \mathbf{m}, \mathbf{S}) = \sum_{i=1}^N \mathbb{E}_{q(f_i)}\left[\log p(y_i \mid f_i)\right] - \mathrm{KL}\left(q(\mathbf{u}) \,\Vert\, p(\mathbf{u})\right)$',
        15, 180
    ),
    'eq4_symbolic.png': (
        r'$\mathcal{L}_{\mathrm{SR}}(g) = \frac{1}{N} \sum_{i=1}^N w_i \left(r_i^{\mathrm{oof}} - g(\mathbf{x}_i)\right)^2 + \lambda \cdot \mathrm{Complexity}(g)$'
        '\n'
        r'$\text{where} \quad w_i = \frac{1}{(\sigma_i^{\mathrm{oof}})^2 + \epsilon}, \qquad r_i^{\mathrm{oof}} = y_i - \mu_i^{\mathrm{oof}}$',
        13, 220
    ),
    'eq5_symbolic.png': (
        r'$\mathcal{L}_{\mathrm{SR}}(g) = \frac{1}{N} \sum_{i=1}^N w_i \left(r_i^{\mathrm{oof}} - g(\mathbf{x}_i)\right)^2 + \lambda \cdot \mathrm{Complexity}(g)$'
        '\n'
        r'$\text{where} \quad w_i = \frac{1}{(\sigma_i^{\mathrm{oof}})^2 + \epsilon}, \qquad r_i^{\mathrm{oof}} = y_i - \mu_i^{\mathrm{oof}}$',
        13, 220
    ),
    'eq5_calibration.png': (
        r'$s^2 = \frac{1}{N_{\mathrm{val}}} \sum_{i=1}^{N_{\mathrm{val}}} \frac{\left(y_i - \hat{y}_i\right)^2}{\sigma_{\mathrm{GP},i}^2}$'
        '\n'
        r'$\mathcal{I}_{95\%}(\mathbf{x}) = \left[ \hat{y}(\mathbf{x}) \pm 1.96 \cdot s \cdot \sigma_{\mathrm{GP}}(\mathbf{x}) \right]$',
        14, 220
    ),
    'eq6_calibration.png': (
        r'$s^2 = \frac{1}{N_{\mathrm{val}}} \sum_{i=1}^{N_{\mathrm{val}}} \frac{\left(y_i - \hat{y}_i\right)^2}{\sigma_{\mathrm{GP},i}^2}$'
        '\n'
        r'$\mathcal{I}_{95\%}(\mathbf{x}) = \left[ \hat{y}(\mathbf{x}) \pm 1.96 \cdot s \cdot \sigma_{\mathrm{GP}}(\mathbf{x}) \right]$',
        14, 220
    ),
    'eq6_discovered.png': (
        r'$\mathbf{CALCE\ CS2\_35:}\quad g^*(\mathbf{x}) = +0.1265 \quad (\text{scalar calibration offset, complexity}=2)$'
        '\n'
        r'$\mathbf{NASA\ B0005:}\quad g^*(\mathbf{x}) = -x_{\mathrm{CCCT}} \cdot x_{\mathrm{IC\_h}} + x_{\mathrm{CCCT}} + 3x_{\mathrm{cycle}} - 5x_{\mathrm{IC\_v}} - \ln\left(|x_{\mathrm{CVCT}}| + 10^{-6}\right) - \sqrt{|x_{\mathrm{IC\_v}} + 0.492|}$',
        12, 240
    ),
    'eq6_discovered_nasa.png': (
        r'$g_{\mathrm{NASA}}^*(\mathbf{x}) = -x_{\mathrm{CCCT}} \cdot x_{\mathrm{IC\_h}} + x_{\mathrm{CCCT}} + 3x_{\mathrm{cycle}}$'
        '\n'
        r'$\qquad\qquad - 5x_{\mathrm{IC\_v}} - \ln\left(|x_{\mathrm{CVCT}}| + 10^{-6}\right) - \sqrt{|x_{\mathrm{IC\_v}} + 0.492|}$',
        13, 220
    ),
    'eq7_discovered_nasa.png': (
        r'$g_{\mathrm{NASA}}^*(\mathbf{x}) = -x_{\mathrm{CCCT}} \cdot x_{\mathrm{IC\_h}} + x_{\mathrm{CCCT}} + 3x_{\mathrm{cycle}}$'
        '\n'
        r'$\qquad\qquad - 5x_{\mathrm{IC\_v}} - \ln\left(|x_{\mathrm{CVCT}}| + 10^{-6}\right) - \sqrt{|x_{\mathrm{IC\_v}} + 0.492|}$',
        13, 220
    ),
    'test_eq_trans.png': (
        r'$\hat{y}(\mathbf{x}) = \mu_{\mathrm{GP}}(\mathbf{x}) + g^*(\mathbf{x}), \qquad \mathbf{x} \in \mathbb{R}^D$',
        17, 170
    ),
}

def render_equation_to_file(latex_code, out_path, fontsize=15, min_h=180, dpi=200, pad_x=50, pad_y=30):
    fig = plt.figure(figsize=(18, 5), dpi=dpi)
    fig.patch.set_alpha(0.0)
    fig.text(0.5, 0.5, latex_code, fontsize=fontsize, color='#0f172a', ha='center', va='center', linespacing=1.6)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.03, transparent=True)
    plt.close(fig)
    
    buf.seek(0)
    raw = Image.open(buf)
    bbox = raw.getbbox()
    if not bbox:
        raise ValueError('No text rendered for ' + latex_code)
        
    cropped = raw.crop(bbox)
    
    card_w = cropped.width + 2 * pad_x
    card_h = max(min_h, cropped.height + 2 * pad_y)
    
    # Solid clean white background (RGB)
    card = Image.new('RGB', (card_w, card_h), (255, 255, 255))
    paste_x = (card_w - cropped.width) // 2
    paste_y = (card_h - cropped.height) // 2
    card.paste(cropped, (paste_x, paste_y), mask=cropped.split()[3])
    
    # Subtle clean border so the card is cleanly bounded on dark-mode viewers
    draw = ImageDraw.Draw(card)
    draw.rectangle([(0, 0), (card_w - 1, card_h - 1)], outline='#cbd5e1', width=2)
    
    card.save(out_path)
    return card.size

def main():
    out_dir = "results/figures/equations"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering {len(EQUATIONS)} equations into {out_dir}...")
    
    for fname, (code, fsz, min_h) in EQUATIONS.items():
        out_path = os.path.join(out_dir, fname)
        size = render_equation_to_file(code, out_path, fontsize=fsz, min_h=min_h, dpi=200)
        print(f"  [OK] {fname:25s} -> {size[0]}x{size[1]} px")
        
    print("\nAll equation images successfully rendered!")

if __name__ == "__main__":
    main()
