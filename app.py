# app.py - Áreas fijas, juegos consistentes, traseras siempre en 2 piezas (original + espejo)
# Requisitos: pip install streamlit pandas matplotlib rectpack
# Ejecutar: streamlit run app.py --server.port 8502

import streamlit as st
import pandas as pd
from datetime import datetime
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

try:
    from rectpack import newPacker
except Exception:
    newPacker = None

st.set_page_config(page_title="Anidado - Final", layout="wide")

# ---------- Config ----------
ROLL_W = 2.0            # ancho del rollo (m)
ROLL_L_MAX = 40.0       # largo máximo (m)
MIN_GAP = 0.004         # gap (m)
COST_M2 = 3.01
SCALE_MM = 1000
COLOR_POLY = "#1f77b4"

# ---------- Plantillas con áreas fijas y puntos reales ----------
POLY_TEMPLATES = {
    # Piloto
    "pilot_a": {
        "pts_px":[
            (132,5),(24,5),(18,12),(12,12),(5,16),(4,325),(17,339),(65,333),(102,333),
            (132,326),(190,328),(218,320),(360,321),(386,326),(450,325),(459,320),
            (458,153),(452,149),(452,142),(444,141),(441,137),(436,137),(428,130),
            (420,130),(408,118),(400,117),(395,113),(396,107),(347,57),(330,34),
            (326,34),(320,29),(244,29),(239,23),(211,21),(197,16),(171,11),(140,11)
        ],
        "w_m":0.5659,"h_m":0.7701,"area":0.3695
    },
    "pilot_b": {
        "pts_px":[
            (143,28),(331,27),(342,39),(342,68),(331,140),(330,177),(337,242),(342,392),
            (344,466),(332,472),(33,471),(26,463),(25,186),(30,152),(30,129),(36,123),(35,118)
        ],
        "w_m":0.6348,"h_m":0.8884,"area":0.5302
    },
    "pilot_c": {
        "pts_px":[
            (35,152),(169,26),(177,26),(182,20),(419,19),(433,31),(432,41),(439,47),
            (426,170),(432,230),(435,549),(426,554),(426,560),(421,566),(24,563),
            (16,556),(15,259),(29,179),(28,164)
        ],
        "w_m":0.6973,"h_m":0.9107,"area":0.5241
    },
    # Copiloto
    "copilot_a": {
        "pts_px":[
            (6,31),(13,20),(18,13),(58,13),(86,20),(146,18),(173,25),(275,25),(282,18),
            (323,18),(330,25),(335,24),(335,61),(348,65),(348,73),(378,72),(384,82),(386,231),
            (375,250),(338,251),(312,257),(258,256),(258,262),(247,268),(247,321),(240,321),
            (234,327),(210,334),(197,333),(164,340),(150,340),(142,345),(76,346),(60,333),
            (60,314),(49,298),(20,298),(11,292),(10,285),(7,284)
        ],
        "w_m":0.5659,"h_m":0.6452,"area":0.2969
    },
    "copilot_b": {
        "pts_px":[
            (92,23),(92,31),(88,35),(85,60),(72,71),(58,71),(33,75),(21,79),(19,89),(36,166),
            (41,430),(57,443),(351,437),(362,427),(362,141),(354,137),(298,137),(292,131),
            (286,131),(274,116),(270,36),(259,24)
        ],
        "w_m":0.6870,"h_m":0.8359,"area":0.4942
    },
    "copilot_c": {
        "pts_px":[
            (125,22),(134,22),(138,15),(349,15),(361,29),(370,28),(374,128),(381,128),(381,134),
            (395,146),(464,154),(480,167),(474,485),(462,497),(462,503),(71,503),(69,497),(59,497),
            (65,214),(41,119),(41,101),(32,80),(45,68),(45,60),(100,62),(114,56),(114,42)
        ],
        "w_m":0.7524,"h_m":0.8146,"area":0.5971
    },
    # Trasera (todas se usarán en 2 piezas: original + espejo)
    "rear_a": {
        "pts_px":[
            (114,8),(196,8),(244,15),(303,14),(303,21),(311,26),(311,84),(316,97),(339,97),
            (360,108),(360,243),(355,258),(346,263),(127,263),(91,269),(67,269),(39,262),
            (17,264),(10,253),(11,156),(29,137),(43,133),(47,121),(47,97),(55,89),(55,85),
            (82,85),(102,74),(102,29)
        ],
        "w_m":0.6039,"h_m":0.4344,"area":0.2172
    },
    "rear_b": {
        "pts_px":[(26,18),(240,16),(239,23),(244,29),(246,332),(235,343),(40,343),(22,333),(22,27)],
        "w_m":0.6533,"h_m":0.4372,"area":0.2895
    },
    "rear_c_1": {
        "pts_px":[(21,85),(34,61),(39,55),(40,37),(54,31),(54,24),(436,25),(436,30),(450,37),
                  (452,297),(448,309),(35,310),(29,306),(28,299),(22,294)],
        "w_m":0.7179,"h_m":0.4784,"area":0.3421
    },
    "rear_c_2": {
        "pts_px":[(21,85),(34,61),(39,55),(40,37),(54,31),(54,24),(436,25),(436,30),(450,37),
                  (452,297),(448,309),(35,310),(29,306),(28,299),(22,294)],
        "w_m":0.7179,"h_m":0.4784,"area":0.3421
    }
}

# ---------- Helpers de datos ----------
def nueva_linea(variant, cantidad, role, meta_extra=None):
    tpl = POLY_TEMPLATES[variant]
    return pd.DataFrame([{
        "role": role,
        "area_m2": tpl["area"],
        "variant": variant,
        "cantidad": int(cantidad),
        "meta": meta_extra or {},
        "creado": datetime.now().isoformat()
    }])

def expand_combinations(df):
    rows=[]
    for _,r in df.iterrows():
        cnt=int(r.get("cantidad",1))
        for _ in range(cnt):
            rows.append({"role":r.get("role",""),"area_m2":r.get("area_m2",0.0),
                         "variant":r.get("variant",""),"meta":r.get("meta",{})})
    return rows

# ---------- Generación de forma real escalada ----------
def generate_shape_coords(variant, x, y, w, h):
    pts_px = POLY_TEMPLATES.get(variant, {}).get("pts_px")
    if not pts_px:
        return {"type":"poly","pts":[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]}
    minx=min(px for px,_ in pts_px); miny=min(py for _,py in pts_px)
    maxx=max(px for px,_ in pts_px); maxy=max(py for _,py in pts_px)
    wpx=maxx-minx; hpx=maxy-miny
    sx=w/float(wpx if wpx>0 else 1); sy=h/float(hpx if hpx>0 else 1)
    pts=[(round((px-minx)*sx+x,6), round((py-miny)*sy+y,6)) for px,py in pts_px]
    return {"type":"poly","pts":pts}

# ---------- Thumbnails (miniaturas reales) ----------
def _template_points_scaled(variant_key, svg_w=160, svg_h=96, pad=6, mirror=False):
    pts_px = POLY_TEMPLATES.get(variant_key, {}).get("pts_px")
    if not pts_px:
        pts = [(pad,pad),(svg_w-pad,pad),(svg_w-pad,svg_h-pad),(pad,svg_h-pad)]
        return pts
    minx=min(px for px,_ in pts_px); miny=min(py for _,py in pts_px)
    maxx=max(px for px,_ in pts_px); maxy=max(py for _,py in pts_px)
    wpx=maxx-minx; hpx=maxy-miny
    s = min((svg_w-2*pad)/(wpx or 1), (svg_h-2*pad)/(hpx or 1))
    offx = pad + (svg_w-2*pad - wpx*s)/2.0
    offy = pad + (svg_h-2*pad - hpx*s)/2.0
    pts=[(round((px-minx)*s+offx,2), round((py-miny)*s+offy,2)) for px,py in pts_px]
    if mirror:
        cx = svg_w/2.0
        pts = [(round(2*cx - x,2), y) for x,y in pts]
    return pts

def make_variant_svg(variant_key, role_key, svg_w=160, svg_h=96):
    mirror = (role_key == "Copiloto")
    if variant_key == "rear_c":  # trasera C se muestra como dos piezas
        pts1 = _template_points_scaled("rear_c_1", svg_w, svg_h, mirror=mirror)
        pts2 = _template_points_scaled("rear_c_2", svg_w, svg_h, mirror=mirror)
        poly1 = " ".join(f"{x},{y}" for x,y in pts1)
        poly2 = " ".join(f"{x},{y}" for x,y in pts2)
        return f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}"><polygon points="{poly1}" fill="#e6f7ff" stroke="#4a6fa3"/><polygon points="{poly2}" fill="#e6f7ff" stroke="#4a6fa3"/></svg>'
    pts = _template_points_scaled(variant_key, svg_w, svg_h, mirror=mirror)
    poly = " ".join(f"{x},{y}" for x,y in pts)
    fill = "#fff2e6" if role_key == "Copiloto" else ("#e6f7ff" if variant_key.startswith("rear") else "#eaeff6")
    return f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}"><polygon points="{poly}" fill="{fill}" stroke="#4a6fa3"/></svg>'

def make_game_svg(game_key, svg_w=200, svg_h=120):
    role_map = {
        "Juego A": ("pilot_a","copilot_a","rear_a"),
        "Juego B": ("pilot_b","copilot_b","rear_b"),
        "Juego C": ("pilot_c","copilot_c","rear_c")
    }
    p,c,r = role_map[game_key]
    svg_p = make_variant_svg(p, "Piloto", svg_w=60, svg_h=60)
    svg_c = make_variant_svg(c, "Copiloto", svg_w=60, svg_h=60)
    svg_r = make_variant_svg(r, "Trasera", svg_w=60, svg_h=60)
    return f'<div style="display:flex;gap:8px;">{svg_p}{svg_c}{svg_r}</div>'

# ---------- Packing (rectpack sin rotación, con gap) ----------
def pack_with_rectpack(pieces, roll_w=ROLL_W, roll_l_max=ROLL_L_MAX, gap=MIN_GAP):
    if newPacker is None:
        raise RuntimeError("rectpack no instalado")
    packer = newPacker(rotation=False)
    packer.add_bin(int(round(roll_l_max*SCALE_MM)), int(round(roll_w*SCALE_MM)))
    for i,p in enumerate(pieces):
        wi_mm = int(round((p["w"]+gap)*SCALE_MM))
        hi_mm = int(round((p["h"]+gap)*SCALE_MM))
        packer.add_rect(wi_mm, hi_mm, rid=i)
    packer.pack()
    placed=[]; not_fit=[]
    for abin in packer:
        for rect in abin:
            rid = rect.rid
            orig = pieces[rid]
            x_m = (rect.x/SCALE_MM) + (gap/2.0)
            y_m = (rect.y/SCALE_MM) + (gap/2.0)
            if x_m + orig["w"] > roll_l_max or y_m + orig["h"] > roll_w:
                not_fit.append(rid); continue
            placed.append({
                "x":round(x_m,6), "y":round(y_m,6),
                "w":round(orig["w"],6), "h":round(orig["h"],6),
                "meta":orig["meta"], "variant":orig["variant"],
                "area":POLY_TEMPLATES[orig["variant"]]["area"]
            })
    return placed, list(set(not_fit))

# ---------- Render principal (sin etiquetas sobre moquetas) ----------
def render_matplotlib(placed, roll_w=ROLL_W):
    fig, ax = plt.subplots(figsize=(11,4.2))
    max_x = max((p["x"]+p["w"]) for p in placed) if placed else 1.0
    ax.set_xlim(0, max(max_x+0.1, 1.0))
    ax.set_ylim(0, roll_w)
    ax.set_aspect('equal', adjustable='box')
    # marco del rollo
    ax.add_patch(MplPolygon([(0,0),(ax.get_xlim()[1],0),
                             (ax.get_xlim()[1],roll_w),(0,roll_w)],
                            closed=True, fill=False, edgecolor='black'))
    for p in placed:
        # polígono real
        polydef = generate_shape_coords(p["variant"], p["x"], p["y"], p["w"], p["h"])
        pts = polydef["pts"]
        if p["meta"].get("mirror", False):
            cx = p["x"] + p["w"]/2.0
            pts = [(round(2*cx - px,6), py) for px,py in pts]
        # dibujar moqueta
        ax.add_patch(MplPolygon(pts, closed=True,
                                facecolor=COLOR_POLY,
                                edgecolor='black', alpha=0.95))
        # bbox referencia
        ax.add_patch(MplPolygon([(p["x"],p["y"]),
                                 (p["x"]+p["w"],p["y"]),
                                 (p["x"]+p["w"],p["y"]+p["h"]),
                                 (p["x"],p["y"]+p["h"])],
                                closed=True, fill=False,
                                edgecolor='grey', linewidth=0.5, alpha=0.25))
    ax.invert_yaxis()
    ax.set_xlabel("Largo (m)")
    ax.set_ylabel("Ancho (m)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf
# ---------- UI ----------
st.title("Demo Anidado de Moquetas Ferplaim CIA Ltda")

if "combinaciones" not in st.session_state:
    st.session_state.combinaciones = pd.DataFrame(columns=["role","area_m2","variant","cantidad","meta","creado"])
if "warn_individual" not in st.session_state:
    st.session_state.warn_individual = False

with st.sidebar:
    st.subheader("Añadir juego completo")
    juego_sel = st.radio("Juego", ["Juego A","Juego B","Juego C"], horizontal=True)
    st.markdown(make_game_svg(juego_sel), unsafe_allow_html=True)
    cant_juego = st.number_input("Número de juegos", min_value=1, step=1, value=1)

    if st.button("Añadir juego"):
        if juego_sel == "Juego A":
            df_add = pd.concat([
                nueva_linea("pilot_a", cant_juego, "Piloto"),
                nueva_linea("copilot_a", cant_juego, "Copiloto", meta_extra={"mirror":True}),
                # Trasera A: original + espejo
                nueva_linea("rear_a", cant_juego, "Trasera"),
                nueva_linea("rear_a", cant_juego, "Trasera", meta_extra={"mirror":True})
            ], ignore_index=True)
        elif juego_sel == "Juego B":
            df_add = pd.concat([
                nueva_linea("pilot_b", cant_juego, "Piloto"),
                nueva_linea("copilot_b", cant_juego, "Copiloto", meta_extra={"mirror":True}),
                # Trasera B: original + espejo
                nueva_linea("rear_b", cant_juego, "Trasera"),
                nueva_linea("rear_b", cant_juego, "Trasera", meta_extra={"mirror":True})
            ], ignore_index=True)
        else:  # Juego C (trasera split en 2 piezas, cada una con espejo)
            df_add = pd.concat([
                nueva_linea("pilot_c", cant_juego, "Piloto"),
                nueva_linea("copilot_c", cant_juego, "Copiloto", meta_extra={"mirror":True}),
                nueva_linea("rear_c_1", cant_juego, "Trasera"),
                nueva_linea("rear_c_1", cant_juego, "Trasera", meta_extra={"mirror":True}),
                nueva_linea("rear_c_2", cant_juego, "Trasera"),
                nueva_linea("rear_c_2", cant_juego, "Trasera", meta_extra={"mirror":True})
            ], ignore_index=True)

        st.session_state.combinaciones = pd.concat([df_add, st.session_state.combinaciones], ignore_index=True)
        st.session_state.warn_individual = False
        st.success("Juego añadido")

    st.markdown("---")
    st.subheader("Añadir individual")
    sel_role = st.radio("Posición", ["Piloto","Copiloto","Trasera"], horizontal=True)

    if sel_role == "Piloto":
        var_map = {"Piloto A":"pilot_a","Piloto B":"pilot_b","Piloto C":"pilot_c"}
    elif sel_role == "Copiloto":
        var_map = {"Copiloto A":"copilot_a","Copiloto B":"copilot_b","Copiloto C":"copilot_c"}
    else:
        var_map = {"Trasera A (2 piezas)":"rear_a","Trasera B (2 piezas)":"rear_b","Trasera C (2 piezas)":"rear_c"}

    sel_label = st.radio("Variante", list(var_map.keys()))
    sel_variant = var_map[sel_label]
    st.markdown(make_variant_svg(sel_variant, sel_role), unsafe_allow_html=True)

    cant_ind = st.number_input("Cantidad", min_value=1, step=1, value=1)

    if st.button("Añadir individual"):
        if sel_variant == "rear_a":
            df_add = pd.concat([
                nueva_linea("rear_a", cant_ind, "Trasera"),
                nueva_linea("rear_a", cant_ind, "Trasera", meta_extra={"mirror":True})
            ], ignore_index=True)
        elif sel_variant == "rear_b":
            df_add = pd.concat([
                nueva_linea("rear_b", cant_ind, "Trasera"),
                nueva_linea("rear_b", cant_ind, "Trasera", meta_extra={"mirror":True})
            ], ignore_index=True)
        elif sel_variant == "rear_c":
            df_add = pd.concat([
                nueva_linea("rear_c_1", cant_ind, "Trasera"),
                nueva_linea("rear_c_1", cant_ind, "Trasera", meta_extra={"mirror":True}),
                nueva_linea("rear_c_2", cant_ind, "Trasera"),
                nueva_linea("rear_c_2", cant_ind, "Trasera", meta_extra={"mirror":True})
            ], ignore_index=True)
        else:
            meta = {"role": sel_role}
            if sel_role == "Copiloto":
                meta["mirror"] = True
            df_add = nueva_linea(sel_variant, cant_ind, sel_role, meta_extra=meta)

        st.session_state.combinaciones = pd.concat([df_add, st.session_state.combinaciones], ignore_index=True)
        st.session_state.warn_individual = True
        st.success("Pieza(s) añadida(s)")

# ---------- Main ----------
st.subheader("Combinación actual")
st.dataframe(st.session_state.combinaciones.reset_index(drop=True), use_container_width=True)

# Advertencia persistente si se añadieron individuales
if st.session_state.warn_individual:
    st.warning("El aprovechamiento puede variar al no ingresar por juegos.")

if st.button("Limpiar combinaciones"):
    st.session_state.combinaciones = pd.DataFrame(columns=["role","area_m2","variant","cantidad","meta","creado"])
    st.session_state.warn_individual = False
    st.success("Combinaciones limpiadas")

st.markdown("---")
st.header("Anidar")

if newPacker is None:
    st.error("rectpack no instalado. Instala: python -m pip install rectpack")

if st.button("Anidar ahora"):
    entries = expand_combinations(st.session_state.combinaciones)
    pieces = []
    for e in entries:
        var = e.get("variant","")
        tpl = POLY_TEMPLATES.get(var)
        if not tpl:
            continue
        w = tpl["w_m"]; h = tpl["h_m"]
        meta = dict(e.get("meta",{}) or {})
        if "role" not in meta:
            meta["role"] = e.get("role","")
        pieces.append({"w":float(w), "h":float(h), "variant":var, "meta":meta})

    try:
        placed, not_fit = pack_with_rectpack(pieces, roll_w=ROLL_W, roll_l_max=ROLL_L_MAX, gap=MIN_GAP)
    except Exception as ex:
        st.error(f"Error en packing: {ex}")
        placed, not_fit = [], list(range(len(pieces)))

    if not placed:
        st.warning("No se ubicó ninguna pieza. Revisa tamaños/gap.")
    else:
        if not_fit:
            st.warning(f"Piezas no ubicadas (IDs): {not_fit}")

        usado_largo = max(p["x"]+p["w"] for p in placed) if placed else 0.0
        total_piece_area = sum(POLY_TEMPLATES[p["variant"]]["area"] for p in placed)
        roll_area_used = ROLL_W * usado_largo if usado_largo > 0 else 0.0
        waste_area = max(0.0, roll_area_used - total_piece_area)
        aprovechamiento = (total_piece_area / roll_area_used) if roll_area_used > 0 else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Largo usado (m)", f"{usado_largo:.3f}")
        c2.metric("Aprovechamiento", f"{aprovechamiento*100:.2f} %")
        c3.metric("Desperdicio (m²)", f"{waste_area:.4f}")
        c4.metric("Pérdida económica (USD)", f"${(waste_area*COST_M2):.2f}")
        c5.metric("Área utilizada (m²)", f"{total_piece_area:.4f}")

        imgbuf = render_matplotlib(placed, roll_w=ROLL_W)
        st.image(imgbuf, caption="Distribución en el rollo", use_container_width=True)

        dfpos = pd.DataFrame([{
            "idx": i+1,
            "role": p["meta"].get("role",""),
            "variant": p.get("variant",""),
            "x": p["x"], "y": p["y"], "w": p["w"], "h": p["h"],
            "area": POLY_TEMPLATES[p["variant"]]["area"]
        } for i,p in enumerate(placed)])

        st.dataframe(dfpos, use_container_width=True)
        st.download_button("Exportar posiciones CSV",
                           data=dfpos.to_csv(index=False),
                           file_name="posiciones_anidado.csv",
                           mime="text/csv")