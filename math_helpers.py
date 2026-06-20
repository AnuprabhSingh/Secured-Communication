"""
OMML (Office Math Markup Language) helpers for python-pptx.

Usage:
    from math_helpers import add_math, M, run_math, frac, sup, sub, supsub, norm, sum_expr

Each helper returns an lxml element you can append to a txBody.
"""
from lxml import etree
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

M_NS  = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A14_NS = "http://schemas.microsoft.com/office/drawing/2010/main"
A_NS  = "http://schemas.openxmlformats.org/drawingml/2006/main"

def _m(tag, **attribs):
    """Create an m: element."""
    el = etree.Element(f"{{{M_NS}}}{tag}")
    for k, v in attribs.items():
        el.set(f"{{{M_NS}}}{k}", v)
    return el

def _r(text, italic=True, bold=False, sz=None):
    """Create an m:r (math run) with text."""
    r = _m("r")
    rpr = _m("rPr")
    if italic:
        sty = _m("sty"); sty.set(f"{{{M_NS}}}val", "i"); rpr.append(sty)
    elif bold:
        sty = _m("sty"); sty.set(f"{{{M_NS}}}val", "b"); rpr.append(sty)
    else:
        sty = _m("sty"); sty.set(f"{{{M_NS}}}val", "p"); rpr.append(sty)
    if sz:
        szEl = _m("sz"); szEl.set(f"{{{M_NS}}}val", str(sz)); rpr.append(szEl)
    r.append(rpr)
    t = _m("t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    return r

def _plain(text):
    """Plain (non-italic) math run."""
    return _r(text, italic=False)

def _sub(base_els, sub_els):
    """m:sSub"""
    s = _m("sSub")
    e = _m("e"); 
    for el in base_els: e.append(el)
    s.append(e)
    sub_el = _m("sub")
    for el in sub_els: sub_el.append(el)
    s.append(sub_el)
    return s

def _sup(base_els, sup_els):
    """m:sSup"""
    s = _m("sSup")
    e = _m("e")
    for el in base_els: e.append(el)
    s.append(e)
    sup_el = _m("sup")
    for el in sup_els: sup_el.append(el)
    s.append(sup_el)
    return s

def _supsub(base_els, sub_els, sup_els):
    """m:sSupSub"""
    s = _m("sSupSub")
    e = _m("e")
    for el in base_els: e.append(el)
    s.append(e)
    sub_el = _m("sub")
    for el in sub_els: sub_el.append(el)
    s.append(sub_el)
    sup_el = _m("sup")
    for el in sup_els: sup_el.append(el)
    s.append(sup_el)
    return s

def _frac(num_els, den_els):
    """m:f (fraction)"""
    f = _m("f")
    num = _m("num")
    for el in num_els: num.append(el)
    den = _m("den")
    for el in den_els: den.append(el)
    f.append(num); f.append(den)
    return f

def _norm(inner_els, begin="|", end="|"):
    """m:d with | delimiters (norm / abs)"""
    d = _m("d")
    dpr = _m("dPr")
    beg = _m("begChr"); beg.set(f"{{{M_NS}}}val", begin); dpr.append(beg)
    en  = _m("endChr"); en.set(f"{{{M_NS}}}val", end);   dpr.append(en)
    d.append(dpr)
    e = _m("e")
    for el in inner_els: e.append(el)
    d.append(e)
    return d

def _paren(inner_els, begin="(", end=")"):
    return _norm(inner_els, begin=begin, end=end)

def _nary(char, sub_els, sup_els, inner_els, limLoc="undOvr"):
    """m:nary — summation, product, etc."""
    n = _m("nary")
    npr = _m("naryPr")
    ch = _m("chr"); ch.set(f"{{{M_NS}}}val", char); npr.append(ch)
    liml = _m("limLoc"); liml.set(f"{{{M_NS}}}val", limLoc); npr.append(liml)
    n.append(npr)
    sub_el = _m("sub")
    for el in sub_els: sub_el.append(el)
    n.append(sub_el)
    sup_el = _m("sup")
    for el in sup_els: sup_el.append(el)
    n.append(sup_el)
    e = _m("e")
    for el in inner_els: e.append(el)
    n.append(e)
    return n

def _rad(inner_els):
    """m:rad (square root)"""
    rad = _m("rad")
    radPr = _m("radPr")
    deg_hide = _m("degHide"); deg_hide.set(f"{{{M_NS}}}val", "1"); radPr.append(deg_hide)
    rad.append(radPr)
    deg = _m("deg"); rad.append(deg)
    e = _m("e")
    for el in inner_els: e.append(el)
    rad.append(e)
    return rad

def _group(inner_els, begin="(", end=")"):
    """Group with over/under – used for angle brackets."""
    return _norm(inner_els, begin=begin, end=end)

def wrap_omath(inner_els, para=True):
    """Wrap elements in m:oMath (and optionally m:oMathPara), then in a14:m."""
    oMath = _m("oMath")
    for el in inner_els:
        oMath.append(el)
    if para:
        oMathPara = _m("oMathPara")
        oMathPara.append(oMath)
        a14m = etree.Element(f"{{{A14_NS}}}m")
        a14m.append(oMathPara)
    else:
        a14m = etree.Element(f"{{{A14_NS}}}m")
        a14m.append(oMath)
    return a14m

def inject_math(txBody, inner_els):
    """Append an OMML math block to a txBody element."""
    a14m = wrap_omath(inner_els)
    txBody.append(a14m)

# ── Pre-built equation blocks ─────────────────────────────────────────────────

def eq_sinr():
    """SINR_k[m] = P_k |h_eff,k^H w_k|^2 / (Σ_{i≠k} P_i|...|^2 + P_{J,k}|h_{J,k}^H z_k|^2 + σ²)"""
    lhs = [
        _supsub([_r("SINR")], [_r("k", italic=True)], []),
        _plain("[m] = ")
    ]
    # numerator: P_k |h_eff,k^H w_k|^2
    num = [
        _sub([_r("P")], [_r("k")]),
        _sup([_norm([
            _supsub([_r("h")], [_r("eff,k")], [_r("H")]),
            _r(" "),
            _sub([_r("w")], [_r("k")])
        ])], [_plain("2")])
    ]
    # denominator: Σ_{i≠k} P_i|...|^2 + P_{J,k}|...|^2 + σ²
    den = [
        _nary("\u2211", [_r("i≠k")], [], [
            _sub([_r("P")], [_r("i")]),
            _sup([_norm([_r("⋯")])], [_plain("2")])
        ]),
        _plain("  +  "),
        _sub([_r("P")], [_r("J,k")]),
        _sup([_norm([
            _supsub([_r("h")], [_r("J,k")], [_r("H")]),
            _r(" "),
            _sub([_r("z")], [_r("k")])
        ])], [_plain("2")]),
        _plain("  +  "),
        _sup([_r("σ")], [_plain("2")])
    ]
    return lhs + [_frac(num, den)]

def eq_rate_sum():
    """R_sum = Σ_k (1/M_sc) Σ_m log2(1 + SINR_k[m])"""
    return [
        _sub([_r("R")], [_r("sum")]),
        _plain(" = "),
        _nary("\u2211", [_r("k")], [], [
            _frac([_plain("1")], [_sub([_r("M")], [_r("sc")])]),
            _plain("  "),
            _nary("\u2211", [_r("m")], [], [
                _r("log"),
                _sub([_plain("")], [_plain("2")]),
                _paren([_plain("1 + "),
                        _supsub([_r("SINR")], [_r("k")], []),
                        _plain("[m]")])
            ])
        ]),
        _plain("   [bits/s/Hz]")
    ]

def eq_protection():
    """η_prot = (1/K·M_sc) Σ_k Σ_m 1{SINR_k[m] ≥ γ_min} × 100%"""
    return [
        _sub([_r("η")], [_r("prot")]),
        _plain(" = "),
        _frac([_plain("1")], [_r("K") , _plain("·"), _sub([_r("M")], [_r("sc")])]),
        _plain("  "),
        _nary("\u2211", [_r("k")], [], [
            _nary("\u2211", [_r("m")], [], [
                _plain("𝟙{"),
                _supsub([_r("SINR")], [_r("k")], []),
                _plain("[m] ≥ "),
                _sub([_r("γ")], [_r("min")]),
                _plain("}")
            ])
        ]),
        _plain(" × 100%")
    ]

def eq_reflection_matrix():
    """Φ[m] = diag(e^{jθ_1}, …, e^{jθ_M})"""
    return [
        _plain("Φ[m] = diag("),
        _sup([_r("e")], [_r("jθ"), _sub([_plain("")], [_plain("1")])]),
        _plain(", … , "),
        _sup([_r("e")], [_r("jθ"), _sub([_plain("")], [_r("M")])]),
        _plain(")")
    ]

def eq_effective_channel():
    """h_eff,k = G^H Φ^H g_{ru,k} + g_{bu,k}"""
    return [
        _sub([_r("h")], [_r("eff,k")]),
        _plain(" = "),
        _sup([_r("G")], [_r("H")]),
        _plain(" "),
        _sup([_r("Φ")], [_r("H")]),
        _plain(" "),
        _sub([_r("g")], [_r("ru,k")]),
        _plain("  +  "),
        _sub([_r("g")], [_r("bu,k")])
    ]

def eq_received_signal():
    """y_k[m] = (g_ru,k^H Φ[m] G[m] + g_bu,k^H) w_k[m] √P_k s_k + interference + noise"""
    return [
        _sub([_r("y")], [_r("k")]),
        _plain("[m]  =  "),
        _paren([
            _supsub([_r("g")], [_r("ru,k")], [_r("H")]),
            _plain(" Φ[m] G[m]  +  "),
            _supsub([_r("g")], [_r("bu,k")], [_r("H")])
        ]),
        _sub([_r("w")], [_r("k")]),
        _plain("[m]  "),
        _rad([_sub([_r("P")], [_r("k")])]),
        _sub([_r("s")], [_r("k")]),
        _plain("  +  interference  +  noise")
    ]

def eq_beamformer():
    """w_k[m] = F_RF f_{BB,k}[m]"""
    return [
        _sub([_r("w")], [_r("k")]),
        _plain("[m]  =  "),
        _sub([_r("F")], [_r("RF")]),
        _plain("  "),
        _sub([_r("f")], [_r("BB,k")]),
        _plain("[m]")
    ]

def eq_mvdr():
    """f_{BB,k} = R_k^{-1} h̃_k / ||R_k^{-1} h̃_k||"""
    return [
        _sub([_r("f")], [_r("BB,k")]),
        _plain("  =  "),
        _frac([
            _sup([_sub([_r("R")], [_r("k")]), _plain("")], [_plain("−1")]),
            _plain("  "),
            _sub([_r("h̃")], [_r("k")])
        ], [
            _norm([
                _sup([_sub([_r("R")], [_r("k")]), _plain("")], [_plain("−1")]),
                _plain("  "),
                _sub([_r("h̃")], [_r("k")])
            ], begin="‖", end="‖")
        ])
    ]

def eq_interference_cov():
    """R_k = σ²I + Σ_{i≠k} P_i h̃_i h̃_i^H"""
    return [
        _sub([_r("R")], [_r("k")]),
        _plain("  =  "),
        _sup([_r("σ")], [_plain("2")]),
        _r("I"),
        _plain("  +  "),
        _nary("\u2211", [_r("i≠k")], [], [
            _sub([_r("P")], [_r("i")]),
            _plain(" "),
            _sub([_r("h̃")], [_r("i")]),
            _supsub([_sub([_r("h̃")], [_r("i")]), _plain("")], [], [_r("H")])
        ])
    ]

def eq_spdp_phase():
    """θ_{q,i}[m] = θ^base_{q,i} + 2π f_m τ_q"""
    return [
        _supsub([_r("θ")], [_r("q,i")], []),
        _plain("[m]  =  "),
        _supsub([_r("θ")], [_r("q,i")], [_plain("base")]),
        _plain("  +  2π "),
        _sub([_r("f")], [_r("m")]),
        _r(" τ"),
        _sub([_plain("")], [_r("q")])
    ]

def eq_beam_squint_gain():
    """η(f_m) = (1/N)|Σ_n exp(j2π n d_s sinθ₀ (f_m/f_c - 1)/λ_c)|²"""
    return [
        _sub([_r("η")], []),
        _paren([_sub([_r("f")], [_r("m")])]),
        _plain("  =  "),
        _frac([_plain("1")], [_r("N")]),
        _sup([
            _norm([
                _nary("\u2211", [_r("n")], [], [
                    _sup([_r("e")], [
                        _plain("j2π n "),
                        _sub([_r("d")], [_r("s")]),
                        _plain(" sinθ₀  "),
                        _frac([_sub([_r("f")], [_r("m")])], [_sub([_r("f")], [_r("c")])]),
                        _sub([_r(" /λ")], [_r("c")])
                    ])
                ])
            ])
        ], [_plain("2")])
    ]

def eq_path_loss():
    """PL(f,d) = (4πfd/c)² · e^{κ_abs(f)·d}"""
    return [
        _plain("PL(f, d)  =  "),
        _sup([_paren([
            _frac([_plain("4π f d")], [_r("c")])
        ])], [_plain("2")]),
        _plain("  ·  "),
        _sup([_r("e")], [
            _sub([_r("κ")], [_r("abs")]),
            _plain("(f) · d")
        ])
    ]

def eq_channel_model():
    """G[m] = √(NM/L) Σ_ℓ α_ℓ(f_m) a_RIS a_BS^H"""
    return [
        _plain("G[m]  =  "),
        _rad([_frac([_r("NM")], [_r("L")])]),
        _plain("  "),
        _nary("\u2211", [_r("ℓ")], [], [
            _sub([_r("α")], [_r("ℓ")]),
            _plain("("),
            _sub([_r("f")], [_r("m")]),
            _plain(")  "),
            _sub([_r("a")], [_r("RIS")]),
            _supsub([_sub([_r("a")], [_r("BS")]), _plain("")], [], [_r("H")])
        ])
    ]

def eq_fuzzy_q():
    """FQ(s,a) = Σ_ℓ ψ_ℓ · Q_ℓ(s,a)"""
    return [
        _plain("FQ(s, a)  =  "),
        _nary("\u2211", [_r("ℓ")], [], [
            _sub([_r("ψ")], [_r("ℓ")]),
            _plain("  ·  "),
            _sub([_r("Q")], [_r("ℓ")]),
            _plain("(s, a)")
        ])
    ]

def eq_q_update():
    """Q_ℓ(s,a) ← Q_ℓ(s,a) + α ψ_ℓ [r + γ max_{a'} FQ(s',a') − Q_ℓ(s,a)]"""
    return [
        _sub([_r("Q")], [_r("ℓ")]),
        _plain("(s, a)  ←  "),
        _sub([_r("Q")], [_r("ℓ")]),
        _plain("(s, a)  +  α "),
        _sub([_r("ψ")], [_r("ℓ")]),
        _plain("  ·  "),
        _paren([
            _r("r"),
            _plain("  +  γ  "),
            _sub([_plain("max"), _plain("")], [_r("a′")]),
            _plain(" FQ(s′, a′)  −  "),
            _sub([_r("Q")], [_r("ℓ")]),
            _plain("(s, a)")
        ])
    ]

def eq_wolf_update():
    """π_ℓ(a*) += ξ ψ_ℓ,   π_ℓ(a) −= ξ ψ_ℓ / (|A|−1)  ∀a≠a*"""
    return [
        _sub([_r("π")], [_r("ℓ")]),
        _paren([_sup([_r("a")], [_plain("∗")])]),
        _plain("  +=  ξ "),
        _sub([_r("ψ")], [_r("ℓ")]),
        _plain(",      "),
        _sub([_r("π")], [_r("ℓ")]),
        _plain("(a)  −=  "),
        _frac([_plain("ξ "), _sub([_r("ψ")], [_r("ℓ")])],
              [_paren([_plain("|𝒜| − 1")])])
    ]

def eq_fuzzy_membership():
    """μ_i(x) = max(0, 1 − |x − c_i| / 0.5),  normalized"""
    return [
        _sub([_r("μ")], [_r("i")]),
        _plain("(x)  =  max"),
        _paren([
            _plain("0,  1  −  "),
            _frac([_norm([_plain("x − "), _sub([_r("c")], [_r("i")])])],
                  [_plain("0.5")])
        ]),
        _plain(",    "),
        _nary("\u2211", [_r("ℓ")], [], [_sub([_r("ψ")], [_r("ℓ")])]),
        _plain(" = 1")
    ]

def eq_jammer_power():
    """P_{J,k} [dBm] = 15 + 0.25·clip(SINR_prev − 5, 0, 20) + 18η + N(0, 1.5²)"""
    return [
        _sub([_r("P")], [_r("J,k")]),
        _plain(" [dBm]  =  15  +  0.25 · clip"),
        _paren([_sub([_r("SINR")], [_r("prev")]),
                _plain(" − 5,  0,  20")]),
        _plain("  +  18η  +  "),
        _r("N"),
        _paren([_plain("0,  1.5²")])
    ]

def eq_predictability():
    """η = 0.25·ρ_rep + 0.35·ρ_dom + 0.40·(1 − H_norm)"""
    return [
        _r("η"),
        _plain("  =  0.25 · "),
        _sub([_r("ρ")], [_r("rep")]),
        _plain("  +  0.35 · "),
        _sub([_r("ρ")], [_r("dom")]),
        _plain("  +  0.40 · "),
        _paren([_plain("1 − "), _sub([_r("H")], [_r("norm")])])
    ]

def eq_feature_pj():
    """f_pj = 0.6·(mean P_J − 15)/25 + 0.4·(max P_J − 15)/25"""
    return [
        _sub([_r("f")], [_r("pj")]),
        _plain("  =  "),
        _frac([_plain("0.6 · "),
               _paren([_plain("mean "),
                       _sub([_r("P")], [_r("J")]),
                       _plain(" − 15")])],
              [_plain("25")]),
        _plain("  +  "),
        _frac([_plain("0.4 · "),
               _paren([_plain("max "),
                       _sub([_r("P")], [_r("J")]),
                       _plain(" − 15")])],
              [_plain("25")])
    ]

def eq_feature_sinr():
    """f_sinr = 0.5·(mean SINR + 10)/40 + 0.5·(min SINR + 10)/40"""
    return [
        _sub([_r("f")], [_r("sinr")]),
        _plain("  =  "),
        _frac([_plain("0.5 · "),
               _paren([_plain("mean SINR + 10")])],
              [_plain("40")]),
        _plain("  +  "),
        _frac([_plain("0.5 · "),
               _paren([_plain("min SINR + 10")])],
              [_plain("40")])
    ]

def eq_reward():
    """r = R_sum − 0.5·P_frac − 3.0·Σ_k 1{SINR_k < γ_min}"""
    return [
        _r("r"),
        _plain("  =  "),
        _sub([_r("R")], [_r("sum")]),
        _plain("  −  0.5 · "),
        _sub([_r("P")], [_r("frac")]),
        _plain("  −  3.0 · "),
        _nary("\u2211", [_r("k")], [], [
            _plain("𝟙{"),
            _sub([_r("SINR")], [_r("k")]),
            _plain(" < "),
            _sub([_r("γ")], [_r("min")]),
            _plain("}")
        ])
    ]

def eq_opt_problem():
    """max_{P_k, Φ[m]} R_sum  s.t. constraints"""
    return [
        _sub([_plain("max")], [_sub([_r("P")], [_r("k")]), _plain(", Φ[m]")]),
        _plain("  "),
        _sub([_r("R")], [_r("sum")]),
        _plain("     s.t.    "),
        _nary("\u2211", [_r("k")], [], [_sub([_r("P")], [_r("k")])]),
        _plain("  ≤  "),
        _sub([_r("P")], [_r("max")]),
        _plain(",    "),
        _sub([_r("SINR")], [_r("k")]),
        _plain("[m]  ≥  "),
        _sub([_r("γ")], [_r("min")]),
        _plain(",    |"),
        _sub([_plain("Φ")], [_r("nn")]),
        _plain("| = 1")
    ]

def eq_wolf_condition():
    """if Σ_a π(a)Q(s,a) > Σ_a π̄(a)Q(s,a): ξ = ξ_win else ξ = ξ_loss"""
    return [
        _plain("If   "),
        _nary("\u2211", [_r("a")], [], [
            _sub([_r("π")], [_r("ℓ")]),
            _plain("(a) · "),
            _sub([_r("Q")], [_r("ℓ")]),
            _plain("(s, a)")
        ]),
        _plain("  >  "),
        _nary("\u2211", [_r("a")], [], [
            _sub([_r("π̄")], [_r("ℓ")]),
            _plain("(a) · "),
            _sub([_r("Q")], [_r("ℓ")]),
            _plain("(s, a)")
        ]),
        _plain(" :   ξ = "),
        _sub([_r("ξ")], [_r("win")]),
        _plain(" = 0.01   else   ξ = "),
        _sub([_r("ξ")], [_r("loss")]),
        _plain(" = 0.04")
    ]


# ── Matplotlib image rendering ───────────────────────────────────────────────
import io as _io

def render_eq(latex_str, fig_w=8.0, fig_h=0.75, fontsize=20,
              bg_color='#F5F8FF', text_color='#1A3A6C'):
    """
    Render a LaTeX math string to a PNG BytesIO using matplotlib mathtext.
    Returns a BytesIO positioned at the start.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(bg_color)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor(bg_color)
    try:
        ax.text(0.5, 0.5, latex_str,
                ha='center', va='center',
                fontsize=fontsize,
                color=text_color,
                transform=ax.transAxes)
    except Exception:
        plain = latex_str.replace('$', '').replace('\\', ' ').replace('{', '').replace('}', '')
        ax.text(0.5, 0.5, plain,
                ha='center', va='center',
                fontsize=fontsize - 2,
                color=text_color,
                transform=ax.transAxes)
    buf = _io.BytesIO()
    try:
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=bg_color, pad_inches=0.08)
    except Exception:
        # Fallback: clear axes, render plain text
        ax.clear(); ax.set_axis_off(); ax.set_facecolor(bg_color)
        plain = latex_str.replace('$', '')
        for cmd in ['\\mathrel', '\\boldsymbol', '\\mathbb', '\\mathop']:
            plain = plain.replace(cmd, '')
        ax.text(0.5, 0.5, plain, ha='center', va='center',
                fontsize=fontsize - 2, color=text_color, transform=ax.transAxes)
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=bg_color, pad_inches=0.08)
    plt.close(fig)
    buf.seek(0)
    return buf
